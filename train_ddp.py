import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from tqdm import tqdm
import time
 
from model.diffusion_model import Diffusion
from diffusion_config import DiffusionConfig
from utils import get_datasets, save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def train(resume_checkpoint_path: str = None):
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = rank == 0
 
    config = DiffusionConfig()
    device = torch.device(f"cuda:{local_rank}")
 
    diffusion_model = Diffusion(config=config).to(device)
    diffusion_model = DDP(diffusion_model, device_ids=[local_rank])
 
    optimiser = AdamW(diffusion_model.parameters(), lr=config.learning_rate)
    scaler = GradScaler()
    lr_scheduler = CosineAnnealingLR(optimiser, T_max=config.epochs, eta_min=1e-6)
 
    start_epoch = 0
    if resume_checkpoint_path:
        # load on every rank, map to this rank's device
        start_epoch = load_checkpoint(resume_checkpoint_path, diffusion_model.module, optimiser)
 
    train_ds, val_ds, _ = get_datasets(config=config, val_split=0.1, seed=42)
 
    # DistributedSampler splits data across ranks, no shuffle=True on the loader itself
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
 
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler,
                               num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, sampler=val_sampler,
                             num_workers=2, pin_memory=True, persistent_workers=True)
 
    if is_main:
        num_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_params:,}")
 
    train_losses, lrs, epoch_times, val_losses = [], [], [], []
 
    for epoch in range(start_epoch, config.epochs):
        start = time.time()
        train_sampler.set_epoch(epoch)  # reshuffle differently each epoch across ranks
        diffusion_model.train()
        train_loss = 0.0
 
        loader = tqdm(train_loader, desc='Train', unit='batch', leave=False) if is_main else train_loader
        for img, _ in loader:
            img = img.to(device)
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
 
        diffusion_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            vloader = tqdm(val_loader, desc='Val', unit='batch', leave=False) if is_main else val_loader
            for img, _ in vloader:
                img = img.to(device)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    loss = diffusion_model(img)
                val_loss += loss.item()
 
        avg_val_loss = val_loss / len(val_loader)
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        epoch_time = time.time() - start
 
        if is_main:
            train_losses.append(avg_loss)
            val_losses.append(avg_val_loss)
            lrs.append(current_lr)
            epoch_times.append(epoch_time)
            tqdm.write(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.2f}s")
 
            if (epoch + 1) % 10 == 0:
                save_checkpoint(diffusion_model.module, optimiser, epoch + 1, avg_loss, avg_val_loss)
 
    if is_main:
        save_checkpoint(diffusion_model.module, optimiser, config.epochs, train_losses[-1], val_losses[-1])
 
    cleanup_ddp()
 
 
if __name__ == "__main__":
    train()