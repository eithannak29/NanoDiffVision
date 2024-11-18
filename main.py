
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from data_loader import MNISTDataModule, CIFAR10DataModule
from model import ViT

if __name__ == "__main__":
    model = ViT()

    mnist_dm = MNISTDataModule()

    cifar10_dm = CIFAR10DataModule()

    logger = TensorBoardLogger("logs", name="ViT")

    trainer = Trainer(
        logger=logger,
        max_epochs=5,
    )
    trainer.fit(model, mnist_dm)
    trainer.test(model, cifar10_dm)
