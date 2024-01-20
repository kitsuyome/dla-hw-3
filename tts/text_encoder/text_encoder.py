from typing import List

import torch
from torch import Tensor

from tts.base.base_text_encoder import BaseTextEncoder
from .text import text_to_sequence


class TextEncoder(BaseTextEncoder):

    def __init__(self, text_cleaners: List[str] = None):
        if text_cleaners is None:
            text_cleaners = ['english_cleaners']
        self.text_cleaners = text_cleaners

    def encode(self, text) -> Tensor:
        encoded = text_to_sequence(text, self.text_cleaners)
        return torch.tensor(encoded)
