from dataclasses import dataclass
import torch

@dataclass
class DiffusionConfig:
    image_size: int = 256
    in_channels: int = 3        # number of color channels (1 = greyscale, 3 = RGB)
    latent_channels: int = 4    # number of l
    base_channels: int = 64     # network width (larger = smarter but slower)
    time_emb_dim: int = 256     # size of time signal vector
    timesteps: int = 1000       # how many steps until image becomes pure static noise
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # noise scheduler (small noise at start, more noise at end)
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # Training: 
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4