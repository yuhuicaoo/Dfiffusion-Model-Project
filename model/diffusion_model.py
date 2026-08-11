import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_config import DiffusionConfig
from model.unet_model import SimpleUNet
from diffusers import AutoencoderKL

class Diffusion(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.model = SimpleUNet(config).to(config.device)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(config.device)

        # noise scheduler of evenly spaced values from beta_start to beta_end for N timesteps
        self.beta = torch.linspace(config.beta_start, config.beta_end, config.timesteps).to(config.device)

        self.alpha = 1 - self.beta
        # cumalative product of alpha up to alpha_t
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)   # (timesteps, )

        # freeze weights for vae model
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.scaling_factor = self.vae.config.scaling_factor

    def noise_images(self, x, t):
        """
        Docstring for noise_images
        
        :param x: batch of clean images, shape: (batch, in_channels, image_size, image_size) or (B,C,H,W)
        :param t: batch of random timesteps, shape: (batch,)

        :returns x_t: batch of noisy images
        :returns noise: batch of pure Gaussian noise used to create x_t
        """

        # reshape to match dimensions of x for broadcasting
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]                 # (batch,) --> (batch, 1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]   # (batch,) --> (batch, 1, 1, 1)

        # generate random noise of shape x
        noise = torch.randn_like(x)             # (batch, in_channels, image_size, image_size)
        
        # return noise aswell for training ground truth
        return (sqrt_alpha_hat * x) + (sqrt_one_minus_alpha_hat * noise), noise
    
    def sample_timesteps(self, n):
        """Generates a batch (of size n) of random timesteps
        
        :param n: batch size
        :return t: batch of random timesteps of size n (batch size)
        """
        return torch.randint(low=0, high=self.config.timesteps, size=(n,), device=self.config.device)

    def _encode(self, images):
        """
        images: expected to be [-1, 1] range
        """
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.scaling_factor

    def _decode(self, latents):
        with torch.no_grad():
            images = self.vae.decode(latents / self.scaling_factor).sample
        return images
    
    def forward(self, x):
        latents = self._encode(x)
        bsz = latents.shape[0]

        # Sample a random batch (size n) of timesteps
        timesteps = self.sample_timesteps(bsz)

        # Create noisy images and get noise used.
        x_t , noise = self.noise_images(latents, timesteps)

        # Predict the noise applied to the original image using a simple U-Net model
        predicted_noise = self.model(x_t, timesteps)

        # Calculte MSE Loss between predicted and actual noise
        return F.mse_loss(noise, predicted_noise)
    
    @torch.no_grad()
    def sample(self, n_samples, steps=50, eta=0.0):
        """Inference step (DDIM): generate images from pure noise"""
        self.model.eval()

        latent_size = self.config.image_size // 8        # latent_size = image_size // 8 (vae downsamples by 8x)
        latents = torch.randn(n_samples, self.config.latent_channels, latent_size, latent_size, device=self.config.device)        # (n_samples, latent_channels, latent_size, latent_size)

        # DDIM timestep schedule
        timesteps = torch.linspace(self.config.timesteps -1, 0, steps, dtype=torch.long, device=self.config.device)

        # loop backwards from t= timesteps-1 to t=1
        for i, timestep in enumerate(timesteps):
            # create timestep tensor
            t = torch.full((n_samples,), timestep, dtype=torch.long, device=self.config.device)

            # get predicted noise from U-Net model
            pred_noise = self.model(latents, t)

            # get noise schedule values for specific timestep
            alpha_hat_t = self.alpha_hat[t][:, None, None, None]

            # previous timestep
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]

                alpha_hat_prev = self.alpha_hat[t_prev].expand(n_samples)[:, None, None, None]
            else:
                # at final step, alpha_hat_prev = 1
                alpha_hat_prev = torch.ones_like(alpha_hat_t)

            # predict x_0
            x0_pred = (latents - torch.sqrt(1 - alpha_hat_t) * pred_noise) / torch.sqrt(alpha_hat_t)
            x0_pred = x0_pred.clamp(-1, 1)

            # DDIM sigma 
            sigma = eta * torch.sqrt((1-alpha_hat_prev) / (1 - alpha_hat_t) * (1 - alpha_hat_t/alpha_hat_prev))

            # direction pointing to x_t
            direction = torch.sqrt(1 - alpha_hat_prev - sigma**2) * pred_noise

            # random noise
            noise = torch.randn_like(latents) if eta > 0 else torch.zeros_like(latents)

            # DDIM update
            latents = torch.sqrt(alpha_hat_prev) * x0_pred + direction + sigma * noise
            
        
        self.model.train()
        # decode latent space representation back to pixel space
        images = self._decode(latents)    
        images = (images.clamp(-1,1) + 1) /2              # [-1, 1] --> [0, 1]
        return images
