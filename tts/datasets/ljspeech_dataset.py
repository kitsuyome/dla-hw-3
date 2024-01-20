import json
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np

import gdown
import librosa
import pyworld as pw
from sklearn.preprocessing import StandardScaler
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
import wget

from tts.base.base_dataset import BaseDataset
from tts.utils import ROOT_PATH, read_json, write_json


logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
    "train_texts": "https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx",
    "mels": "https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j",
    "alignments": "https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip"
}


class LJspeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir

        if not (self._data_dir / 'downloaded').exists():
            self._load_dataset()
        if not (self._data_dir / 'preprocessed').exists():
            self._preprocess_dataset()
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        if not (self._data_dir / 'wavs').exists():
            arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
            print(f"Loading LJSpeech")
            download_file(URL_LINKS["dataset"], arch_path)
            shutil.unpack_archive(arch_path, self._data_dir)
            for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
                shutil.move(str(fpath), str(self._data_dir / fpath.name))
            os.remove(str(arch_path))
            shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

        # download train texts
        if not (self._data_dir / 'train.txt').exists():
            gdown.download(URL_LINKS['train_texts'],
                           str(self._data_dir / 'train.txt'))

        # download mels
        if not (self._data_dir / 'mels').exists():
            arch_path = str(self._data_dir / 'mel.tar.gz')
            gdown.download(URL_LINKS['mels'], arch_path)
            shutil.unpack_archive(arch_path, self._data_dir)
            os.remove(arch_path)

        # download alignments
        if not (self._data_dir / 'alignments').exists():
            arch_path = str(self._data_dir / 'alignments.zip')
            wget.download(URL_LINKS['alignments'], out=arch_path)
            shutil.unpack_archive(arch_path, self._data_dir)
            os.remove(arch_path)

        '''
        Content of the self._data_dir:
            - wavs
            - train.txt
            - mels
            - alignments
        '''

        Path(self._data_dir / 'downloaded').touch()

    def _preprocess_dataset(self):
        # list all files of dataset
        wav_paths = [file_name for file_name in sorted((self._data_dir / "wavs").iterdir())]
        with open(self._data_dir / 'train.txt') as f:
            texts = [text[:-1] for text in f]
        mel_paths = [file_name for file_name in sorted((self._data_dir / "mels").iterdir())]
        duration_paths = [str(self._data_dir / "alignments" / f"{i}.npy") for i in range(len(wav_paths))]

        # prepare directories for pitch and energy
        (self._data_dir / 'pitch').mkdir(exist_ok=True)
        (self._data_dir / 'energy').mkdir(exist_ok=True)

        index = []
        pitch_min, pitch_max, energy_min, energy_max = math.inf, -math.inf, math.inf, -math.inf
        pitch_scaler, energy_scaler = StandardScaler(), StandardScaler()
        for i in tqdm(range(len(wav_paths)), desc='Preparing pitches and energies'):
            data_dict = {
                'audio_path': str(wav_paths[i]),
                'mel_target_path': str(mel_paths[i]),
                'duration_path': str(duration_paths[i]),
                'text': texts[i]
            }

            # get pitch
            wav, sr = librosa.load(data_dict['audio_path'])
            pitch, t = pw.dio(wav.astype(np.float64), sr, frame_period=256 / sr * 1000)
            pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)

            # save pitch and update stats
            path = str(self._data_dir / 'pitch' / f'{i}.npy')
            np.save(path, pitch)
            data_dict['pitch_path'] = path
            pitch_min = min(pitch_min, np.min(pitch))
            pitch_max = max(pitch_max, np.max(pitch))
            pitch_scaler.partial_fit(pitch.reshape((-1, 1)))

            # get energy
            mel = np.load(data_dict['mel_target_path'])
            energy = np.linalg.norm(mel, axis=1)

            # save energy and update stats
            path = str(self._data_dir / 'energy' / f'{i}.npy')
            np.save(path, energy)
            data_dict['energy_path'] = path
            energy_min = min(energy_min, np.min(energy))
            energy_max = max(energy_max, np.max(energy))
            energy_scaler.partial_fit(energy.reshape((-1, 1)))

            index.append(data_dict)

        pitch_mean, pitch_std = pitch_scaler.mean_[0], pitch_scaler.scale_[0]
        self._normalize(str(self._data_dir / 'pitch'), pitch_mean, pitch_std)

        energy_mean, energy_std = energy_scaler.mean_[0], energy_scaler.scale_[0]
        self._normalize(str(self._data_dir / 'energy'), energy_mean, energy_std)

        # save stats
        stats = {
            'pitch_min': float((pitch_min - pitch_mean) / pitch_std),
            'pitch_max': float((pitch_max - pitch_mean) / pitch_std),
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),

            'energy_min': float((energy_min - energy_mean) / energy_std),
            'energy_max': float((energy_max - energy_mean) / energy_std),
            'energy_mean': float(energy_mean),
            'energy_std': float(energy_std),
        }
        write_json(stats, self._data_dir / 'stats.json')

        # save index for test and dev
        train_length = int(0.9 * len(index))  # hand split, dev ~ 10%
        write_json(index[:train_length], self._data_dir / 'train_index.json')
        write_json(index[train_length:], self._data_dir / 'dev_index.json')

        Path(self._data_dir / 'preprocessed').touch()

    def _normalize(self, in_dir, mean, std):
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

    def _get_or_load_index(self, part):
        if part == 'train':
            index_path = self._data_dir / "train_index.json"
        else:
            index_path = self._data_dir / "dev_index.json"

        with index_path.open() as f:
            index = json.load(f)
        return index

    def get_stats(self):
        return read_json(self._data_dir / 'stats.json')
