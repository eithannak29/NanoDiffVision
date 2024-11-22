# NanoDiffVision 🐜

NanoDiffVision is a research project in Deep Learning aimed at re-implementing the Vision Transformer (ViT) from scratch while exploring and comparing two attention mechanisms: the classic Self-Attention and Differential Attention (introduced in the DIFF Transformer paper). The project focuses on compact models, making the approach accessible and efficient even with limited resources.

## 📚 Background

Transformers have revolutionized the field of computer vision. NanoDiffVision focuses on re-implementing the Vision Transformer (ViT) while evaluating Differential Attention, a recent mechanism designed to reduce attention noise and focus more on relevant information. The project builds on the insights from the Vision Transformer paper ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) and the [DIFF Transformer paper](https://arxiv.org/abs/2410.05258), which introduces the Differential Attention mechanism.

## 🎯 Objectives

- **Reproduce the Vision Transformer (ViT)** from scratch to better understand its components.
- **Integrate and compare** classic Self-Attention with Differential Attention.
- **Analyze the performance** of compact models, especially in terms of their ability to handle relevant information while limiting noise.

## 🛠️ Installation

To install and use NanoDiffVision, follow the steps below:

```bash
git clone https://github.com/eithannak29/NanoDiffVision.git
cd NanoDiffVision
uv sync
```

## 🧪 Usage

To train the model, use the provided Makefile:

To train on all CIFAR10 configurations:

```bash
make cifar10
```

To train on all MNIST configurations:

```bash
make mnist
```

To train on all FashionMNIST configurations:

```bash
make fashionmnist
```

To run a specific configuration file:

```bash
make config CONFIG_FILE=configs/CIFAR10/lite_config.yml
```

## 📊 Results

## 🏛️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
