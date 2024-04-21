import torch
import torch.nn.functional as F
import config
# from scheduler import betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, get_values_at_timesteps
from scheduler import get_values_at_timesteps

@torch.no_grad()
def sample_image_at_timestemp_minus_one(
  model,
  images_at_timesteps, # x_t of (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
  timestep, # t of (1)
):

  # TODO: Get rid of duplication here and in scheduler; it is caused because we need to fix initialization due to multiple GPUs
  betas = torch.linspace(
    0.0001,
    0.02,
    config.T,
  )
  alphas = 1.0 - betas
  alphas_cumprod = torch.cumprod(alphas, axis = 0)
  alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.0)
  sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
  # sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) # TODO: I do not think it is ever used
  sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
  posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

  beta_t = get_values_at_timesteps(
    betas, # (T)
    timestep, # (1)
  ) # (1, 1, 1, 1)

  sqrt_one_minus_alphas_cumprod_t = get_values_at_timesteps(
    sqrt_one_minus_alphas_cumprod,
    timestep,
  )

  sqrt_recip_alphas_t = get_values_at_timesteps(
    sqrt_recip_alphas,
    timestep,
  )

  noises_at_timestep_predicted = model(
    images_at_timesteps,
    timestep,
  ) # epsilon_t

  images_at_timesteps_minus_one = sqrt_recip_alphas_t * (images_at_timesteps - beta_t * noises_at_timestep_predicted / sqrt_one_minus_alphas_cumprod_t)

  posterior_variance_t = get_values_at_timesteps(
    posterior_variance,
    timestep,
  )

  if timestep == 0:

    return images_at_timesteps_minus_one

  else:

    noises = torch.randn_like(images_at_timesteps)

    return images_at_timesteps_minus_one + torch.sqrt(posterior_variance_t) * noises

@torch.no_grad()
def sample_image(
  model,
):

  # image = x_T
  image = torch.randn(
    (
      # 2, # batch_size = 1
      1,
      config.IMG_CHANNELS,
      config.IMG_SIZE,
      config.IMG_SIZE,
    )
  ) # (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

  images_at_timesteps = torch.zeros(config.T, *image.shape) # (T, BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

  for i in range(0, config.T)[:: -1]:

    # TODO: this most likely requires movement from CPU to MPS; See if it is possible to change it somehow
    timestep = torch.tensor(
      [i],
      dtype = torch.int64,
    )

    # TODO: Do not override image but collect
    image = sample_image_at_timestemp_minus_one(
      model,
      image,
      timestep,
    )

    image = torch.clamp(image, -1.0, 1.0) #TODO: What does it do?

    images_at_timesteps[i] = image

  return images_at_timesteps