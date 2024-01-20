import logging

from tts.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomTextDataset(BaseDataset):
    def __init__(self, texts_file, *args, **kwargs):
        index = []
        with open(texts_file) as f:
            for i, text in enumerate(f):
                index.append(
                    {
                        'text': text[:-1],
                        'text_number': i     # for test.py
                    }
                )

        super().__init__(index, *args, **kwargs)
