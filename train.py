import torch
import time
from model.diffusion_model import Diffusion
from diffusion_config import DiffusionConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import get_dataloaders, save_checkpoint, load_checkpoint
from torch.amp import GradScaler


def train(resume_checkpoint_path: str = None):
    config = DiffusionConfig()
    diffusion_model = Diffusion(config=config).to(config.device)
    optimiser = AdamW(diffusion_model.parameters(), lr=config.learning_rate)
    scaler = GradScaler()
    lr_scheduler = CosineAnnealingLR(optimiser, T_max = config.epochs, eta_min = 1e-6)

    start_epoch = 0
    if resume_checkpoint_path:
        start_epoch = load_checkpoint(
            resume_checkpoint_path, diffusion_model, optimiser
        )

    train_loader, val_loader, _ = get_dataloaders(config=config, val_split=0.2, seed=42)
    print(len(train_loader), len(val_loader))

    num_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    train_losses, lrs, epoch_times, val_losses = [], [], [], []

    for epoch in range(start_epoch, config.epochs):
        start = time.time()
        print(f"Epoch {epoch + 1}/{config.epochs}")
        diffusion_model.train()
        train_loss = 0.0


        # training
        for img, _ in tqdm(train_loader, desc='Train', unit='batch', leave=False):
            img = img.to(config.device)

            optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = diffusion_model(img)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        train_losses.append(avg_loss)

        # validation
        diffusion_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for img, _ in tqdm(val_loader, desc='Val', unit='batch', leave=False):
                img = img.to(config.device)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    loss = diffusion_model(img)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        lrs.append(current_lr)
        tqdm.write(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.2f}s")

        # save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(diffusion_model, optimiser, epoch + 1, avg_loss, avg_val_loss)

    save_checkpoint(diffusion_model, optimiser, config.epochs, train_losses[-1], val_losses[-1])

    return {
        "train_losses": train_losses,
        'val_losses': val_losses,
        "learning_rates": lrs,
        "epoch_timnes": epoch_times,
    }
