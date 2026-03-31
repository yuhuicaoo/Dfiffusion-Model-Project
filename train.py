import torch
import time
from model.diffusion_model import Diffusion
from diffusion_config import DiffusionConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import get_dataloader, save_checkpoint, load_checkpoint
from torch.amp import GradScaler


def train(resume_checkpoint_path: str = None):
    config = DiffusionConfig()
    diffusion_model = Diffusion(config=config).to(config.device)
    optimiser = AdamW(diffusion_model.parameters(), lr=config.learning_rate)
    scaler = GradScaler()

    start_epoch = 0
    if resume_checkpoint_path:
        start_epoch = load_checkpoint(
            resume_checkpoint_path, diffusion_model, optimiser
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

            optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = diffusion_model(images)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            epoch_loss += loss.item()

            # if step % 100 == 0:
            #     avg = epoch_loss / (step + 1)
            #     print(
            #         f"Epoch {epoch+1}/{config.epochs} | Step {step}/{len(dataloader)} | Loss {avg:.4f}"
            #     )

        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - start
        all_loss.append(avg_loss)
        all_lr.append(config.learning_rate)
        all_epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1} finished | Avg loss {avg_loss:.4f}")

        # save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(diffusion_model, optimiser, epoch + 1, avg_loss)
        
    save_checkpoint(diffusion_model, optimiser, config.epochs, all_loss[-1])
    return all_loss, all_lr, all_epoch_times


if __name__ == "__main__":
    train()
