import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from tts.base.base_text_encoder import BaseTextEncoder
from tts.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            text_encoder: BaseTextEncoder,
            config_parser: ConfigParser,
            limit=None,
            max_audio_length=None,
            max_text_length=None,
    ):
        self.text_encoder = text_encoder
        self.config_parser = config_parser

        # index = self._filter_records_from_dataset(index, max_audio_length, max_text_length, limit)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        sample = data_dict.copy()
        if 'audio_path' in data_dict:
            sample['audio'] = self.load_audio(data_dict['audio_path'])  # [1 x T]
        if 'mel_target_path' in data_dict:
            sample['mel_target'] = torch.from_numpy(np.load(data_dict['mel_target_path']))  # [L' x freq]
        if 'duration_path' in data_dict:
            sample['duration_target'] = torch.from_numpy(np.load(data_dict['duration_path']))  # [L]
        if 'pitch_path' in data_dict:
            sample['pitch_target'] = torch.from_numpy(np.load(data_dict['pitch_path']))  # [L']
        if 'energy_path' in data_dict:
            sample['energy_target'] = torch.from_numpy(np.load(data_dict['energy_path']))  # [L']
        sample['text'] = data_dict['text']  # str
        sample['text_encoded'] = self.text_encoder.encode(data_dict['text'])  # [L]

        return sample

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        '''
        Load audio from path, resample it if needed
        :return: 1st channel of audio (tensor of shape 1xL)
        '''
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def get_stats(self):
        raise NotImplementedError()

    def _filter_records_from_dataset(
            self, index: list, max_audio_length, max_text_length, limit
    ) -> list:
        '''
        Filter records depending on max_audio_length, max_text_length and limit
        '''
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_len"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if max_text_length is not None:
            exceeds_text_length = (
                    np.array(
                        [len(self.text_encoder.encode(el["text"])) for el in index]
                    )
                    >= max_text_length
            )
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            # random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index
