import torch
from model.diffusion_model import Diffusion
from diffusion_config import DiffusionConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

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
        num_workers=0,
        pin_memory=False
    )

def train():
    config=DiffusionConfig()
    diffusion_model = Diffusion(config=config).to(config.device)
    optimiser = AdamW(diffusion_model.parameters(), lr=config.learning_rate)
    lr_scheduler = CosineAnnealingLR(optimizer=optimiser, T_max=config.epochs)
    dataloader = get_dataloader(config=config)

    num_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    for epoch in tqdm(range(config.epochs)):
        diffusion_model.train()
        epoch_loss = 0.0

        for step, (images, _) in enumerate(dataloader):
            images = images.to(config.device)

            optimiser.zero_grad()

            loss = diffusion_model(images)
            loss.backward()

            optimiser.step()

            epoch_loss += loss.item()

            if step % 100 == 0:
                avg = epoch_loss / (step + 1)
                print(f"Epoch {epoch+1}/{config.epochs} | Step {step}/{len(dataloader)} | Loss {avg:.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished | Avg loss {avg_loss:.4f}")

    

if __name__ == "__main__":
    train()