import torch
import torch.nn.functional as F

from tts.base.base_metric import BaseMetric

class DurationMSE(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, log_duration_pred, duration_target, src_pad_mask, **batch):
        log_duration_target = torch.log(duration_target.float() + 1.0)
        return F.mse_loss(log_duration_pred, log_duration_target)

class EnergyMSE(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, energy_pred, energy_target, **batch):
        return F.mse_loss(energy_pred, energy_target)

class MelMSE(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, mel_pred, mel_target, **batch):
        return F.mse_loss(mel_pred, mel_target)

class PitchMSE(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, pitch_pred, pitch_target, **batch):
        return F.mse_loss(pitch_pred, pitch_target)
