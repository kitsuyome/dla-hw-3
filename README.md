# Text-to-Speech

## Objective

In this project explored the task of speech synthesis by implementing the FastSpeech2 model. The model was developed based on the paper ["FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"](https://arxiv.org/pdf/2006.04558.pdf). For training the model, the [LJSpeech dataset](http://keithito.com/LJ-Speech-Dataset) was used.

An attempt was made to reproduce the results of the research and achieve high-quality synthesized speech, which was quite successful. Experiments were also conducted in speech synthesis with different pitches, speeds, and energies. The test audios are included in the Wandb report.

## Report

[Wandb link](https://api.wandb.ai/links/kitsuyomi/3jgyk8uk)

## Installation

Clone the repository and install dependencies:

```
!git clone https://github.com/kitsuyome/dla-hw-3
%cd dla-hw-3
!pip install -r requirements.txt
```

## Test

Run the setup script to download the model checkpoint, test the model, and generate WAV audio files from the texts in the 'test_data' directory.

```
!python setup.py
!python test.py \
  -c test_data/config.json \
  -r test_data/checkpoint.pth \
  -t test_data/texts.txt \
  -o test_data
```

## Reproduce Training

Run the script to reproduce training:

```
!python train.py -c tts/configs/config.json
```

## Test audio

To explore the generated audio with various configurations (alterations in pitch, speed, and energy), please refer to the [Wandb report](https://api.wandb.ai/links/kitsuyomi/3jgyk8uk)

## License

[MIT License](LICENSE)
