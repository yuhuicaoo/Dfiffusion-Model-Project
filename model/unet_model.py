import math
import torch
import torch.nn as nn
from diffusion_config import DiffusionConfig

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Creates a embedding vector from a timestep t through a series of sin() and cos()
        functions with different progressing frequencies
        """
        device = time.device
        # split embedding into half, half for sin() and half for cos()
        half_dim = self.dim // 2
        # create frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # each timestep gets multiplied by all frequencies
        embeddings = time[:, None] * embeddings[None, :]  # (B, half_dim)
        # apply sin() and cos()
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)  # (B, dim)
        return embeddings


class Block(nn.Module):
    """Simple Conv --> GroupNorm --> GELU block"""

    def __init__(self, in_channels, out_channels, time_embedding_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_embedding_dim, out_channels)

        # Decoder - upsampling
        if up:
            # padding = (kernel_size - 1) / 2
            # first convolution must handle 2*in_channels due to concatenation of skip connections
            self.conv1 = nn.Conv2d(
                2 * in_channels, out_channels, kernel_size=3, padding=1
            )
            self.transform = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        # Encoder - downsampling
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bnorm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.bnorm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.GELU()

    def forward(self, x, t):
        # first convolution
        h = self.bnorm1(self.relu(self.conv1(x)))

        time_embeddding = self.relu(self.time_mlp(t))
        time_embeddding = time_embeddding[(...,) + (None,) * 2]
        h = h + time_embeddding
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SimpleUNet(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        in_channels = config.in_channels
        output_dim = config.in_channels
        time_emb_dim = config.time_emb_dim
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
        )

        self.conv0 = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)
        self.downs = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1], time_emb_dim)
                for i in range(len(down_channels) - 1)
            ]
        )
        self.ups = nn.ModuleList(
            [
                Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True)
                for i in range(len(up_channels) - 1)
            ]
        )
        self.output = nn.Conv2d(up_channels[-1], output_dim, kernel_size=1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # concatenate skip connection to inputs
            x = torch.cat((x, residual_x), dim=-1)
            x = up(x, t)
        return self.output(x)
