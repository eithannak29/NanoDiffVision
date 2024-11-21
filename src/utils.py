import yaml
from typing import Dict, Any
from data_loader import MNISTDataModule, FashionMNISTDataModule, CIFAR10DataModule


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_data_module(config_data: Dict[str, Any]):
    data_name = config_data.pop("name", None)
    if data_name == "MNIST":
        return MNISTDataModule(**config_data)
    elif data_name == "FashionMNIST":
        return FashionMNISTDataModule(**config_data)
    elif data_name == "CIFAR10":
        return CIFAR10DataModule(**config_data)
    else:
        raise ValueError(f"Invalid dataset name: {data_name}")
