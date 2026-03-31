import torch
import time
from model.diffusion_model import Diffusion
from diffusion_config import DiffusionConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import get_dataloader, save_checkpoint, load_checkpoint


def train(resume_checkpoint_path: str = None):
    config = DiffusionConfig()
    diffusion_model = Diffusion(config=config).to(config.device)
    optimiser = AdamW(diffusion_model.parameters(), lr=config.learning_rate)
    lr_scheduler = CosineAnnealingLR(optimizer=optimiser, T_max=config.epochs)

    start_epoch = 0
    if resume_checkpoint_path:
        start_epoch = load_checkpoint(
            resume_checkpoint_path, diffusion_model, optimiser, lr_scheduler
        )

    dataloader = get_dataloader(config=config)

    num_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    all_loss = []
    all_lr = []
    all_epoch_times = []

    for epoch in tqdm(range(start_epoch, config.epochs), desc="Epochs"):
        start = time.time()
        diffusion_model.train()
        epoch_loss = 0.0

        for _, (images, _) in enumerate(dataloader):
            images = images.to(config.device)

            optimiser.zero_grad()

            loss = diffusion_model(images)
            loss.backward()

            optimiser.step()

            epoch_loss += loss.item()

            # if step % 100 == 0:
            #     avg = epoch_loss / (step + 1)
            #     print(
            #         f"Epoch {epoch+1}/{config.epochs} | Step {step}/{len(dataloader)} | Loss {avg:.4f}"
            #     )

        lr_scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - start
        all_loss.append(avg_loss)
        all_lr.append(lr_scheduler.get_last_lr()[0])
        all_epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1} finished | Avg loss {avg_loss:.4f}")

        # save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(diffusion_model, optimiser, lr_scheduler, epoch + 1, avg_loss)
        
    save_checkpoint(diffusion_model, optimiser, lr_scheduler, config.epochs, all_loss[-1])
    return all_loss, all_lr, all_epoch_times


if __name__ == "__main__":
    train()
