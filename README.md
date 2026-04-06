# Dfiffusion Model Project

A personal learning project where I built and trained a diffusion model from scratch using PyTorch, trained on CIFAR-10 dataset.

## Table of Contents

- [Purpose](#-purpose)
- [Overview](#-overview)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#project-structure)
- [Key Configurations](#key-configurations-diffusion_configpy)
- [Contact](#-contact)

## 🎯 Purpose

I got curious about how image generation actually works under the hood, not just calling an API, but understanding the actual math and architecture behind it. So I decided to research, learn, and build a diffusion model from scratch as a way to learn by doing. This project is my attempt at understanding the forward noising process, training a UNet to predict the noise, the reverse denoising process,

Still a work in progress and very much a learning experience!

## 🧠 Overview

A diffusion model works in two phases:

- **Forward process:** Gradually add Gaussian noise to the image over _t_ timesteps until it becomes pure static.
- **Teaching process:** Train a neural network (UNet) to learn the noise that was added to the image.
- **Reverse process:** Generate new images from pure static noise by applying the learned UNet.

## 🛠️ Tech Stack

| Component     | Technology                 |
| :------------ | :------------------------- |
| Deep Learning | PyTorch                    |
| Dataset       | CIFAR-10 (via torchvision) |
| Model         | Custom UNet                |
| Language      | Python 3.10+               |

## 📁Project Structure

```
├── Diffusion-Model-Project/    # Root directory
│   ├── train.py                # Script for training loop
│   ├── diffusion_config.py     # Configurration for hyperparameters
│   ├── utils.py                # DataLoader, checkpoint save & load
│   └── model/
│       ├── diffusion_model.py  # Diffusion Model architecture
|       └── unet_model.py       # U-Net architecture
```

## Key Configurations (`diffusion_config.py`)

```python
image_size   = 256     # input image resolution
timesteps    = 1000    # noising steps
base_channels = 128    # network width
batch_size   = 16
epochs       = 100
learning_rate = 1e-4
beta_start = 1e-4
beta_end: = 0.02
```

##  🔮 Next steps
- [ ] Experiment with different beta schedules (cosine vs linear)
- [ ] Train diffusion model on CelebA or custom dataset
- [ ] Research how to improve image quality
- [ ] Implement cross-attention mechanism for conditional image generation


## 📬 Contact

Made by **Yuhui Cao** — feel free to reach out!

- **GitHub**: [yuhuicaoo](https://github.com/yuhuicaoo)
- **LinkedIn**: [Yuhui Cao](https://www.linkedin.com/in/yuhuicao/)
- **Email**: yuhuicao20@gmail.com