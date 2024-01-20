import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **batch):
        loss = F.mse_loss(batch['mel_pred'], batch['mel_target'])
        if 'duration_target' in batch:
            log_duration_target = torch.log(batch['duration_target'].float() + 1.0)
            loss += F.mse_loss(batch['log_duration_pred'], log_duration_target)
        if 'pitch_target' in batch:
            loss += F.mse_loss(batch['pitch_pred'], batch['pitch_target'].float())
        if 'energy_target' in batch:
            loss += F.mse_loss(batch['energy_pred'], batch['energy_target'].float())
        return loss
