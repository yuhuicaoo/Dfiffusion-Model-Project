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
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(config.device)

        # noise scheduler of evenly spaced values from beta_start to beta_end for N timesteps
        self.beta = torch.linspace(config.beta_start, config.beta_end, config.timesteps).to(config.device)

        self.alpha = 1 - self.beta
        # cumalative product of alpha up to alpha_t
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)   # (timesteps, )

        # freeze weights for vae model
        for params in self.vae.parameters():
            params.requires_grad = False

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
        return torch.randint(low=1, high=self.config.timesteps, size=(n,), device=self.config.device)
    
    def forward(self, x):
        # Sample a random batch (size n) of timesteps
        timesteps = self.sample_timesteps(x.shape[0])

        # encode pixel representation into latent space representation
        with torch.no_grad():
            latent_representation = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

        # Create noisy images and get noise used.
        x_t , noise = self.noise_images(latent_representation, timesteps)

        # Predict the noise applied to the original image using a simple U-Net model
        predicted_noise = self.model(x_t, timesteps)

        # Calculte MSE Loss between predicted and actual noise
        return F.mse_loss(noise, predicted_noise)
    
    @torch.no_grad()
    def sample(self, n_samples):
        """Inference step: generate images from pure noise"""

        self.model.eval()
        latent_size = self.config.image_size // 8        # latent_size = image_size // 8 (vae downsamples by 8x)
        x = torch.randn((n_samples, self.config.latent_channels, latent_size, latent_size)).to(self.config.device)        # (n_samples, latent_channels, latent_size, latent_size)

        # loop backwards from t= timesteps-1 to t=1
        for i in reversed(range(1, self.config.timesteps)):
            t = (torch.ones(n_samples) * i).long().to(self.config.device)

            # get predicted noise from U-Net model
            predicted_noise = self.model(x, t)

            # get noise schedule values for specific timestep
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            # add new Gaussian noise only if t > 1, else zeros if t = 1
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) *noise
        
        # decode latent space representation back to pixel space
        x = x / self.vae.config.scaling_factor
        x = self.vae.decode(x).sample

    
        self.model.train()
        # scale from [-1, 1] to [0, 1]
        x = (x.clamp(-1,1) + 1) /2
        return x
