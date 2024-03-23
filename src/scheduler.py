import torch
import torch.nn.functional as F
import config

# Evenly spaced values from `start` to `end`
betas = torch.linspace(
  0.0001,
  0.02,
  config.T,
)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis = 0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

def get_values_at_timesteps(
  values_at_timesteps, # of (T)
  timesteps, # t of (BATCH_SIZE)
):

  """
  We want to get `value`(s) at timestep `t`
  """

  return values_at_timesteps.gather(
    -1,
    timesteps,
  ).reshape(
    timesteps.shape[0], # BATCH_SIZE
    1,
    1,
    1,
  )

def images_to_images_with_noise_at_timesteps(
  images, # x_0 of (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
  timesteps, # t of (BATCH_SIZE)
):

  """
  Takes an image and a timestep as input and return the noisy version of it
  """

  # We sample noise from `normal` distribution
  # We used `_like` to generate noise value for each value of `x_0`
  noises = torch.randn_like(images) # this is our epsilon; (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

  sqrt_alphas_cumprod_at_timestep = get_values_at_timesteps(
    sqrt_alphas_cumprod,
    timesteps,
  ) # (BATCH_SIZE, 1, 1, 1)

  sqrt_one_minus_alphas_cumprod_at_timestep = get_values_at_timesteps(
    sqrt_one_minus_alphas_cumprod,
    timesteps,
  ) # (BATCH_SIZE, 1, 1, 1)

  # Reparametrization trick
  images_with_noise = sqrt_alphas_cumprod_at_timestep * images + sqrt_one_minus_alphas_cumprod_at_timestep * noises # (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

  return images_with_noise, noises
