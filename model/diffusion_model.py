import torch
import torch.nn as nn
from diffusion_config import DiffusionConfig
from model.unet_model import SimpleUNet


class Diffusion(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.model = SimpleUNet(config).to(config.device)

        # noise scheduler of evenly spaced values from beta_start to beta_end for N timesteps
        self.beta = torch.linspace(config.beta_start, config.beta_end, config.timesteps).to(config.device)

        self.alpha = 1 - self.beta
        # cumalative product of alpha up to alpha_t
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)   # (timesteps, )

    def noise_images(self, x, t):
        """
        Docstring for noise_images
        
        :param x: batch of clean images, shape: (batch, in_channels, image_size, image_size) or (B,C,H,W)
        :param t: batch of random timesteps, shape: (batch,)
        """

        # reshape to match dimensions of x for broadcasting
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]                 # (batch,) --> (batch, 1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]   # (batch,) --> (batch, 1, 1, 1)

        # generate random noise of shape x
        noise = torch.randn_like(x)             # (batch, in_channels, image_size, image_size)
        
        # return noise aswell for training ground truth
        return (sqrt_alpha_hat * x) + (sqrt_one_minus_alpha_hat * noise), noise