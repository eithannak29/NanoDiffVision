import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from data_loader import MNISTDataModule, CIFAR10DataModule
from utils import load_config
from model import ViT


def main(config_path):
    config = load_config(config_path)

    config_data = config["data"]
    config_model = config["model"]
    config_trainer = config["trainer"]

    if config["data"]["name"] == "MNIST":
        data_module = MNISTDataModule(
            data_dir=config_data["data_dir"],
            batch_size=config_data["batch_size"],
            num_workers=config_data["num_workers"],
        )
    elif config["data"]["name"] == "CIFAR10":
        data_module = CIFAR10DataModule(
            data_dir=config_data["data_dir"],
            batch_size=config_data["batch_size"],
            num_workers=config_data["num_workers"],
        )
    else:
        raise ValueError("Invalid dataset name")

    model = ViT(
        in_channels=config_model["in_channels"],
        image_size=config_model["image_size"],
        patch_size=config_model["patch_size"],
        embedding_dim=config_model["embedding_dim"],
        hidden_dim=config_model["hidden_dim"],
        num_blocks=config_model["num_blocks"],
        n_heads=config_model["n_heads"],
        out_dim=config_model["out_dim"],
        dropout=config_model["dropout"],
        use_diff_attention=config_model["use_diff_attention"],
    )

    logger = WandbLogger(
        save_dir=config_trainer["logger"]["save_dir"],
        name=config_trainer["logger"]["name"],
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=config_trainer["max_epochs"],
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
