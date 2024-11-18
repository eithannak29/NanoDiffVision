import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/MNIST",
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers
        )


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data/CIFAR10", batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [45000, 5000], generator=torch.Generator().manual_seed(42)
            )
        if stage == "test":
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.test_transform
            )

        if stage == "predict":
            self.cifar_predict = CIFAR10(
                self.data_dir, train=False, transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.cifar_predict, batch_size=self.batch_size, num_workers=self.num_workers)
