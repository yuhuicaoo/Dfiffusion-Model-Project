import torch
import torch.nn as nn


class SelfAttention2D(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # flatten spatial dims for attention
        h = self.norm(x)
        h = h.reshape(B, C, H*W).transpose(1, 2)    # (B, H*W, C)

        # every spatial position attends to each other
        h, _ = self.attn(h, h, h)

        # projection
        h = self.proj_out(h)

        # reshape back
        h = h.transpose(1, 2).reshape(B, C, H, W)       # (B, C, H, W)

        # residual connection
        return x + h