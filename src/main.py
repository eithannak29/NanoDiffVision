import os
import argparse
from typing import Dict, Any
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import load_config, get_data_module
from vit import ViT


def train_model(config: Dict[str, Any]):
    data_module = get_data_module(config["data"])
    model = ViT(**config["model"], **config["trainer"])
    logger = WandbLogger(**config["logger"])

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=config["save"]["dir"],
        filename=config["save"]["name"] + "-{epoch:02d}-{val_loss:.4f}",
    )

    trainer = Trainer(
        logger=logger, callbacks=[checkpoint_callback], **config["trainer"]
    )
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)

    save_path = os.path.join(config["save"]["dir"], config["save"]["name"])
    trainer.save_checkpoint(save_path)


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
