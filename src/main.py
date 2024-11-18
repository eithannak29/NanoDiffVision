import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from data_loader import MNISTDataModule, CIFAR10DataModule
from utils import load_config
from model import ViT


def main(config_path):
    config = load_config(config_path)

    if config["data"]["name"] == "MNIST":
        data_module = MNISTDataModule(
            data_dir=config["data"]["data_dir"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
        )
    elif config["data"]["name"] == "CIFAR10":
        data_module = CIFAR10DataModule(
            data_dir=config["data"]["data_dir"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
        )
    else:
        raise ValueError("Invalid dataset name")

    model = ViT(
        in_channels=config["model"]["in_channels"],
        shape=config["model"]["shape"],
        patch_size=config["model"]["patch_size"],
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        n_blocks=config["model"]["n_blocks"],
        n_heads=config["model"]["n_heads"],
        out_dim=config["model"]["out_dim"],
        dropout=config["model"]["dropout"],
    )

    logger = WandbLogger(
        save_dir=config["trainer"]["logger"]["save_dir"],
        name=config["trainer"]["logger"]["name"],
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=config["trainer"]["max_epochs"],
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)
    trainer.save_checkpoint("model.ckpt")


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
