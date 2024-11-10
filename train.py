import os
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

def load_data():
    return CIFAR10(os.getcwd(), download=True, transform=ToTensor())


if __name__ == "__main__":
    data = load_data()
