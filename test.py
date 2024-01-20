import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import tts.model as module_model
import tts.text_encoder as module_text_encoder
import tts.vocoder as module_vocoder
from tts.trainer import Trainer
from tts.utils import ROOT_PATH
from tts.utils.object_loading import get_dataloaders
from tts.utils.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "test_data" / "checkpoint.pth"

def main(config, out_dir):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    text_encoder = config.init_obj(config['text_encoder'], module_text_encoder)

    # vocoder
    vocoder = config.init_obj(config['vocoder'], module_vocoder)

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)

            output = model(**batch, use_target_duration=False)
            for i in range(len(output["mel_pred"])):
                mel_pred = output["mel_pred"][i].unsqueeze(0).transpose(1, 2)
                length = torch.sum(output["duration_pred_rounded"][i])
                mel_pred = mel_pred[:, :, :length]
                torchaudio.save(
                    Path(out_dir) / f'{batch["text_number"][i]}.wav',
                    torch.tensor(vocoder.inference(mel_pred)).unsqueeze(0),
                    config["preprocessing"]["sr"]
                )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        type=str,
        help="Directory to write results (.wav files)",
    )
    args.add_argument(
        "-t",
        "--test-texts-file",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-texts-file` was provided, set it as a default test set
    if args.test_texts_file is not None:
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomTextDataset",
                        "args": {
                            "texts_file": args.test_texts_file
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
