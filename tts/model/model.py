import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tts.utils.util import create_alignment, get_non_pad_mask, get_attn_key_pad_mask
from tts.base.base_model import BaseModel


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class VariancePredictor(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.input_size = model_config['hidden_size']
        self.filter_size = model_config['variance_predictor_filter_size']
        self.dropout = model_config['variance_predictor_dropout']

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(self.input_size, self.filter_size, kernel_size=3, padding=1),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(self.filter_size, self.filter_size, kernel_size=3, padding=1),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.filter_size, 1)

    def forward(self, x):
        x = self.conv_net(x)
        out = self.linear_layer(x)
        out = out.squeeze(-1)
        return out


class LengthRegulator(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.pad_id = model_config['pad_id']
        self.duration_predictor = VariancePredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        # expand_max_len = torch.max(
        #     torch.sum(duration_predictor_output, -1), -1)[0]
        expand_max_len = torch.max(torch.sum(duration_predictor_output, -1)).item()
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, src_pos, alpha=1.0, duration_target=None, use_target_duration=True):
        duration_predictor_output = self.duration_predictor(x)

        duration_rounded = (torch.exp(duration_predictor_output) - 1) * alpha
        duration_rounded = torch.round(torch.clamp(duration_rounded, min=0)).int()

        if use_target_duration:
            assert duration_target is not None
            x = self.LR(x, duration_target)
        else:
            x = self.LR(x, duration_rounded)
        return duration_predictor_output, duration_rounded, x


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config, stats):
        super().__init__()

        self.pad_id = model_config['pad_id']
        self.use_target_pitch_and_energy = model_config.get('use_target_pitch_and_energy', True)

        self.length_regulator = LengthRegulator(model_config)
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.d_control = model_config.get('duration_control', 1.0)
        self.p_control = model_config.get('pitch_control', 1.0)
        self.e_control = model_config.get('energy_control', 1.0)

        n_bins = model_config["n_bins"]
        pitch_min = stats["pitch_min"]
        pitch_max = stats["pitch_max"]
        energy_min = stats["energy_min"]
        energy_max = stats["energy_max"]

        self.pitch_bins = nn.Parameter(
            torch.linspace(pitch_min, pitch_max, n_bins - 1),
            requires_grad=False,
        )

        self.energy_bins = nn.Parameter(
            torch.linspace(energy_min, energy_max, n_bins - 1),
            requires_grad=False,
        )

        self.pitch_embedding = nn.Embedding(n_bins, model_config["hidden_size"])
        self.energy_embedding = nn.Embedding(n_bins, model_config["hidden_size"])

    def get_pitch_embedding(self, x, target, mask):
        prediction = self.pitch_predictor(x)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * self.p_control
            embedding = self.pitch_embedding(torch.bucketize(prediction, self.pitch_bins))
        return prediction, embedding * mask

    def get_energy_embedding(self, x, target, mask):
        prediction = self.energy_predictor(x)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * self.e_control
            embedding = self.energy_embedding(torch.bucketize(prediction, self.energy_bins))
        return prediction, embedding * mask

    def forward(
        self,
        x,
        src_pos,
        mel_pos,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        use_target_duration=True
    ):

        (
            log_duration_prediction,
            duration_rounded,
            x
        ) = self.length_regulator(x, src_pos, self.d_control, duration_target, use_target_duration)

        if not use_target_duration:
            # recalculate mel_pos
            lengths = duration_rounded.sum(1)
            mel_pos = torch.full((len(x), max(lengths)), self.pad_id, dtype=torch.int, device=lengths.device)
            for i, length in enumerate(lengths):
                mel_pos[i, :length] = torch.arange(1, length + 1, dtype=torch.int, device=lengths.device)

        mask = get_non_pad_mask(mel_pos, self.pad_id)
        if use_target_duration:
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, mask)
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, mask)
        else:
            # pass None target to use predicted pitch and energy
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, None, mask)
            energy_prediction, energy_embedding = self.get_energy_embedding(x, None, mask)
        x = x + pitch_embedding + energy_embedding

        return (
            x,
            log_duration_prediction,
            duration_rounded,
            pitch_prediction,
            energy_prediction,
            mel_pos
        )

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # q, k, v: [ (batch_size * n_heads) x seq_len x hidden_size ]

        attn = torch.bmm(q, k.transpose(1, 2)).masked_fill(mask, -torch.inf)
        attn = self.softmax(attn / self.temperature)

        # attn: [ (batch_size * n_heads) x seq_len x seq_len ]

        output = torch.bmm(attn, v)

        # output: [ (batch_size * n_heads) x seq_len x hidden_size ]

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
         # normal distribution initialization better than kaiming(default in pytorch)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_v)))

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, fft_conv1d_kernel, fft_conv1d_padding, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=fft_conv1d_kernel[0], padding=fft_conv1d_padding[0])
        # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=fft_conv1d_kernel[1], padding=fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class FFTBlock(torch.nn.Module):
    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, fft_conv1d_kernel, fft_conv1d_padding, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.pad_id = model_config['pad_id']

        self.src_word_emb = nn.Embedding(model_config['vocab_size'], model_config['hidden_size'],
                                         padding_idx=self.pad_id)
        self.position_enc = nn.Embedding(model_config['max_seq_len'] + 1, model_config['hidden_size'],
                                         padding_idx=self.pad_id)

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config['hidden_size'],
            model_config['fft_conv1d_filter_size'],
            model_config['num_heads'],
            model_config['hidden_size'] // model_config['num_heads'],
            model_config['hidden_size'] // model_config['num_heads'],
            model_config['fft_conv1d_kernel'],
            model_config['fft_conv1d_padding'],
            dropout=model_config['dropout']
        ) for _ in range(model_config['num_layers'])])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(src_seq, src_seq, self.pad_id)
        non_pad_mask = get_non_pad_mask(src_seq, self.pad_id)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output


class Decoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.pad_id = model_config['pad_id']

        self.position_enc = nn.Embedding(model_config['max_seq_len'] + 1, model_config['hidden_size'],
                                         padding_idx=self.pad_id)

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config['hidden_size'],
            model_config['fft_conv1d_filter_size'],
            model_config['num_heads'],
            model_config['hidden_size'] // model_config['num_heads'],
            model_config['hidden_size'] // model_config['num_heads'],
            model_config['fft_conv1d_kernel'],
            model_config['fft_conv1d_padding'],
            dropout=model_config['dropout']
        ) for _ in range(model_config['num_layers'])])

    def forward(self, enc_seq, mel_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(mel_pos, mel_pos, self.pad_id)
        non_pad_mask = get_non_pad_mask(mel_pos, self.pad_id)

        # -- Forward
        dec_output = enc_seq + self.position_enc(mel_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list
        else:
            return dec_output

class FastSpeech2(BaseModel):
    def __init__(self, stats, **model_config):
        super().__init__()

        self.pad_id = model_config['pad_id']

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config, stats)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config['hidden_size'], model_config['n_mels'])

    def forward(self,
                text_encoded,
                src_pos,
                mel_pos=None,
                pitch_target=None,
                energy_target=None,
                duration_target=None,
                use_target_duration=True,
                **batch):

        encoder_output = self.encoder(text_encoded, src_pos)

        (
            va_output,
            log_duration_pred,
            duration_pred_rounded,
            pitch_pred,
            energy_pred,
            mel_pos
        ) = self.variance_adaptor(encoder_output, src_pos, mel_pos, pitch_target,
                                  energy_target, duration_target, use_target_duration)

        decoder_output = self.decoder(va_output, mel_pos)

        mel_pred = self.mel_linear(decoder_output)

        mel_non_pad_mask = get_non_pad_mask(mel_pos, self.pad_id)
        return {
            'mel_pred': mel_pred * mel_non_pad_mask,
            'log_duration_pred': log_duration_pred * src_pos.ne(self.pad_id),
            'duration_pred_rounded': duration_pred_rounded * src_pos.ne(self.pad_id),
            'pitch_pred': pitch_pred * mel_non_pad_mask.squeeze(-1),
            'energy_pred': energy_pred * mel_non_pad_mask.squeeze(-1),
            'src_pad_mask': src_pos.eq(self.pad_id)
        }
