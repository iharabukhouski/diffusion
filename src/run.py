#! /usr/bin/env python3

import os

os.system('clear')

import torch
import config
import device
from scheduler import betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, get_values_at_timesteps
from checkpoints import restore_weights
from model import UNet
from plt import plt_images

@torch.no_grad()
def sample_image_at_timestemp_minus_one(
  model,
  images_at_timesteps, # x_t of (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
  timestep, # t of (1)
):

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
      2, # batch_size = 1
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

model = UNet()

restore_weights(
  model,
)

model.eval()

images = sample_image(model)

plt_images(images)
