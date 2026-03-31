from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusion_config import DiffusionConfig
from model.diffusion_model import Diffusion
import os
import torch

def get_dataloader(config: DiffusionConfig):

    transform = transforms.Compose([
        # resize images to be compatible with model
        transforms.Resize((config.image_size, config.image_size)),
        # converts images to [0, 1] scale from [0, 255]
        transforms.ToTensor(),
        # normalise images to [-1, 1] scale
        transforms.Normalize([0.5] * config.in_channels, [0.5] * config.in_channels)
    ])

    # intialise dataset
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )


def save_checkpoint(model, optimiser, epoch, loss, output_dir: str = "checkpoints"):
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/checkpoint_epoch_{epoch:04d}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "loss": loss,
    }, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(path, model, optimiser):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    print(f"Resumed from {path} (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    return checkpoint["epoch"] + 1