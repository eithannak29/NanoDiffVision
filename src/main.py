import os
import argparse
from typing import Dict, Any
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from utils import load_config, get_data_module
from vit import ViT


def train_model(config: Dict[str, Any]):
    data_module = get_data_module(config["data"])
    model = ViT(**config["model"], **config["trainer"])
    enabled_logger = config["logger"].pop("enabled", None)
    logger = None
    if enabled_logger: 
        logger = WandbLogger(**config["logger"])
    trainer = Trainer(logger=logger, **config["trainer"])
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)


def main(config_path: str):
    try:
        config = load_config(config_path)
        train_model(config)
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Vision Transformer model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    main(args.config)
