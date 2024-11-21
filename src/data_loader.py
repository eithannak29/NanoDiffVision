import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir, batch_size, num_workers, transform, test_transform=None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.test_transform = test_transform or transform

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class MNISTDataModule(BaseDataModule):
    def __init__(self, data_dir="./data/MNIST", batch_size=128, num_workers=4):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        super().__init__(data_dir, batch_size, num_workers, transform)

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage in ("fit", None):
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.train_dataset, self.val_dataset = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )
        if stage in ("test", "predict", None):
            self.test_dataset = MNIST(
                self.data_dir, train=False, transform=self.transform
            )
            self.predict_dataset = self.test_dataset


class FashionMNISTDataModule(BaseDataModule):
    def __init__(self, data_dir="./data/FashionMNIST", batch_size=128, num_workers=4):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        super().__init__(data_dir, batch_size, num_workers, transform)

    def prepare_data(self):
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage in ("fit", None):
            fashion_mnist_full = FashionMNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.train_dataset, self.val_dataset = random_split(
                fashion_mnist_full,
                [55000, 5000],
                generator=torch.Generator().manual_seed(42),
            )
        if stage in ("test", "predict", None):
            self.test_dataset = FashionMNIST(
                self.data_dir, train=False, transform=self.transform
            )
            self.predict_dataset = self.test_dataset


class CIFAR10DataModule(BaseDataModule):
    def __init__(self, data_dir="./data/CIFAR10", batch_size=128, num_workers=4):
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=10),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        super().__init__(data_dir, batch_size, num_workers, transform, test_transform)

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage in ("fit", None):
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.train_dataset, self.val_dataset = random_split(
                cifar_full, [45000, 5000], generator=torch.Generator().manual_seed(42)
            )
        if stage in ("test", "predict", None):
            self.test_dataset = CIFAR10(
                self.data_dir, train=False, transform=self.test_transform
            )
            self.predict_dataset = self.test_dataset
