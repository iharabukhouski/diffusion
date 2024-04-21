#! /usr/bin/env python3

# NOTE: just run it as `./test_forward_pass.py`

T = 100
BATCH_SIZE = 1

import torch
from data import create_dataloader
from scheduler import images_to_images_with_noise_at_timesteps
from plt import plt_images
from logger import Logger
from functools import partial

_logger = partial(Logger, 0)

dataloader = create_dataloader(
  _logger,
  'cpu',
  0,
  1,
)

image = next(iter(dataloader))[0]

images_at_timesteps = torch.zeros(T, *image.shape) # (T, BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

images_at_timesteps[0] = image

for i in range(1, T):

  timesteps = torch.tensor([i], dtype = torch.int64)

  image_with_noise, noise = images_to_images_with_noise_at_timesteps(image, timesteps)

  images_at_timesteps[i] = image_with_noise

plt_images(images_at_timesteps)
