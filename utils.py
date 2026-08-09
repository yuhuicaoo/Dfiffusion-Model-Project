from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from diffusion_config import DiffusionConfig
from model.diffusion_model import Diffusion
import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationsWrapper:
    def __init__(self, albumentations_transform):
        self.transform = albumentations_transform

    def __call__(self, img):
        img = np.array(img)
        return self.transform(image=img)["image"]

def build_train_val_transforms(config: DiffusionConfig) -> tuple[AlbumentationsWrapper, AlbumentationsWrapper]:
    train_transform = AlbumentationsWrapper(A.Compose([
        A.Resize(config.image_size, config.image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
        A.Normalize(mean=[0.5] * config.in_channels, std=[0.5] * config.in_channels),
        ToTensorV2()
    ]))

    eval_transform = AlbumentationsWrapper(A.Compose([
        A.Resize(config.image_size, config.image_size),
        A.Normalize(mean=[0.5] * config.in_channels, std=[0.5] * config.in_channels),
        ToTensorV2()
    ]))

    return train_transform, eval_transform

def split_train_val_indices(n_total: int, val_split: float = 0.1, seed: int = 42):
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_indices, val_indices = torch.utils.data.random_split(
        range(n_total), [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    return train_indices.indices, val_indices.indices

def get_datasets(config: DiffusionConfig, val_split: float = 0.1, seed: int = 42):
    train_transform, eval_transform = build_train_val_transforms(config=config)

    raw = datasets.CIFAR10(root="./data", train=True, download=False)
    train_indices, val_indices = split_train_val_indices(len(raw), val_split, seed)

    train_ds_full = datasets.CIFAR10(root="./data", train=True, transform=train_transform, download=False)
    val_ds_full = datasets.CIFAR10(root="./data", train=True, transform=eval_transform, download=False)
    test_ds = datasets.CIFAR10(root='./data', train=False, transform=eval_transform, download=False)

    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(val_ds_full, val_indices)
    return train_ds, val_ds, test_ds


def get_dataloaders(config: DiffusionConfig, val_split: float = 0.1, seed: int = 42):
    train_ds, val_ds, test_ds = get_datasets(config, val_split, seed)

    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader


def save_checkpoint(model, optimiser, epoch, train_loss, val_loss, output_dir: str = "checkpoints"):
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/checkpoint_epoch_{epoch:04d}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss
    }, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(path, model, optimiser):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    print(f"Resumed from {path} (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    return checkpoint["epoch"] + 1