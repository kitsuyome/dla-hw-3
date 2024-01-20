import logging
from typing import List

import numpy as np
import torch

from tts.utils.util import pad_1D_tensor, pad_2D_tensor

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    '''
    Collates batch the format:
    {
        'text': list[str],                        always
        'audio': list[tensor[1, T]],

        'text_encoded': tensor[B, max(L)],        always
        'duration_target': tensor[B, max(L)],
        'src_pos': tensor[B, max(L)],             always

        'mel_target': tensor[B, max(L'), freq],
        'pitch_target': tensor[B, max(L')],
        'energy_target': tensor[B, max(L')],
        'mel_pos': tensor[B, max(L')]
    }
    '''
    result_batch = dict()

    if 'text_number' in dataset_items[0]:
        result_batch['text_number'] = [item["text_number"] for item in dataset_items]

    result_batch['text'] = [item["text"] for item in dataset_items]
    if 'audio' in dataset_items[0]:
        result_batch["audio"] = [item["audio"] for item in dataset_items]
    if 'mel_target' in dataset_items[0]:
        result_batch['mel_target'] = pad_2D_tensor([item["mel_target"] for item in dataset_items])

    for key in ['text_encoded', 'duration_target', 'pitch_target', 'energy_target']:
        if key in dataset_items[0]:
            result_batch[key] = pad_1D_tensor([item[key] for item in dataset_items])

    # add src_pos
    length_text = np.array([len(item['text_encoded']) for item in dataset_items])
    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i + 1 for i in range(int(length_src_row))],
                              (0, max_len - int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))
    result_batch['src_pos'] = src_pos

    # add mel_pos
    if 'mel_target' in dataset_items[0]:
        mel_targets = [item["mel_target"] for item in dataset_items]
        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.size(0))

        mel_pos = list()
        max_mel_len = int(max(length_mel))
        for length_mel_row in length_mel:
            mel_pos.append(np.pad([i + 1 for i in range(int(length_mel_row))],
                                  (0, max_mel_len - int(length_mel_row)), 'constant'))
        mel_pos = torch.from_numpy(np.array(mel_pos))
        result_batch['mel_pos'] = mel_pos

    return result_batch
