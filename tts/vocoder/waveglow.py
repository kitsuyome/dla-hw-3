import os

import gdown
from scipy.io.wavfile import write
import torch

import tts.vocoder.glow as glow
from tts.utils.util import ROOT_PATH


MAX_WAV_VALUE = 32768.0


def inference(mel, waveglow, sigma):
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=sigma)
        audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    # write(audio_path, sampling_rate, audio)
    return audio


def get_WaveGlow(waveglow_path):
    # waveglow_path = os.path.join("waveglow", "pretrained_model")
    # waveglow_path = os.path.join(waveglow_path, "waveglow_256channels.pt")
    wave_glow = torch.load(waveglow_path)['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.cuda().eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')
    return wave_glow


class WaveGlow:
    def __init__(self):
        pretrain_dir = ROOT_PATH / 'tts' / 'vocoder' / 'pretrained_model'
        ckpt_path = pretrain_dir / 'waveglow_256channels.pt'
        if not pretrain_dir.exists():
            os.mkdir(str(pretrain_dir))
            gdown.download('https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx', str(ckpt_path))

        self.model = get_WaveGlow(str(ckpt_path))

    def inference(self, mel, speed=1.0):
        '''
        :param mel: tensor[1, freq, L']
        :param speed: float
        :return: np.array[T]
        '''
        return inference(mel, self.model, speed)
