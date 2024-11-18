from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from data_loader import MNISTDataModule, CIFAR10DataModule
from model import ViT

if __name__ == "__main__":
    dm = CIFAR10DataModule()

    model = ViT(in_channels=3, patch_size=4, embedding_dim=384, hidden_dim=1536, n_blocks=7, n_heads=6, out_dim=10, dropout=0.1)

    logger = TensorBoardLogger("logs", name="ViT")

    trainer = Trainer(
        logger=logger,
        max_epochs=50,
    )
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)
