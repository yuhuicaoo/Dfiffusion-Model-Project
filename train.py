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

    print("Precomputing latents")
    all_latents = []
    vae = diffusion_model.vae
    vae.eval()

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Encoding"):
            images = images.to(config.device)
            latents = (
                vae.encode(images.float()).latent_dist.sample()
                * vae.config.scaling_factor
            )
            all_latents.append(latents.cpu())

    all_latents = torch.cat(all_latents)
    torch.save(all_latents, "latents.pt")

    del diffusion_model.vae
    torch.cuda.empty_cache()

    latent_dataset = torch.utils.data.TensorDataset(all_latents)
    latent_dataloader = torch.utils.data.DataLoader(
        latent_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    for epoch in tqdm(range(start_epoch, config.epochs), desc="Epoch"):
        start = time.time()
        print(f"Epoch {epoch + 1}/{config.epochs}")
        diffusion_model.train()
        epoch_loss = 0.0

        for _, (latents, _) in enumerate(latent_dataloader):
            latents = latents.to(config.device)

            optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = diffusion_model(latents)
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
        all_loss.append(avg_loss)

        epoch_time = time.time() - start
        all_epoch_times.append(epoch_time)
        print(
            f"Epoch {epoch + 1} finished | Training Loss {avg_loss:4f} \n \
              Epoch Time: {epoch_time:.2f}s | Avg Epoch Time: {(sum(all_epoch_times) / len(all_epoch_times)):.2f}s"
        )

        all_lr.append(config.learning_rate)

        # save checkpoint every 10 epochs
        if (epoch + 1) % 50 == 0:
            save_checkpoint(diffusion_model, optimiser, epoch + 1, avg_loss)

    save_checkpoint(diffusion_model, optimiser, config.epochs, all_loss[-1])
    return {
        "losses": all_loss,
        "learning_rates": all_lr,
        "epoch_timnes": all_epoch_times,
    }


if __name__ == "__main__":
    train()
