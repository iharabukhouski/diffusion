#! /usr/bin/env python3

import os

os.system('clear')

import time
import torch
from torch.optim import Adam
import torch.nn.functional as F
from data import create_dataloader, print_dataloader
import config
import logger
import device
import wandb
from model import UNet, print_model
from scheduler import images_to_images_with_noise_at_timesteps
from checkpoints import restore_weights, save_weights, save_architecture

def calculate_loss(
  model,
  images, # x_0 of (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
  timesteps, # t of (BATCH_SIZE)
):

  # image_with_noise_at_timestep of (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
  # noises of (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
  images_with_noise_at_timesteps, noises = images_to_images_with_noise_at_timesteps(
    images,
    timesteps,
  )

  noises_predicted = model(
    images_with_noise_at_timesteps,
    timesteps,
  )

  return F.l1_loss(
    noises,
    noises_predicted,
  )

def train(
  dataloader,
  model,
  optimizer,
  step,
):

  for epoch in range(config.NUMBER_OF_EPOCHS):

    # images / x_0 of (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    for batch, (images, labels) in enumerate(dataloader):

      optimizer.zero_grad()

      if device.DEVICE == device.MPS:

        # NOTE: Due to MPS fallback???
        images = images.to(device.MPS)

      elif device.DEVICE == device.CUDA:

        # TODO: I do not know why we need it but without this the device is CPU on a machine with CUDA
        images = images.to(device.CUDA)

      # timesteps of (BATCH_SIZE)
      timesteps = torch.randint(
        0, # low
        config.T, # high
        (
            config.BATCH_SIZE,
        ), # size
        dtype = torch.int64,
      )

      loss = calculate_loss(
        model,
        images,
        timesteps,
      )
      loss_number = loss.item()

      loss.backward()

      optimizer.step()

      # if batch % 10 == 0:

      logger.info(f'Step: {step:>4} | Epoch: {epoch:>4} | Batch: {batch:>4} | Loss: {loss_number}')

      # accuracy = 1

      wandb.log(
        {
          # 'accuracy': accuracy,
          'loss': loss_number,
        },
        step = step,
      )

      step += 1

    logger.info('Loss', loss_number)

  return step, loss_number


dataloader = create_dataloader()

print_dataloader(dataloader)

model = UNet()

wandb.watch(
  models = model,
  log = 'all',
  log_freq = 1,
  log_graph = True,
)

print_model(model)

optimizer = Adam(
  model.parameters(),
  lr = config.LEARNING_RATE,
)

step = restore_weights(
  model,
  optimizer,
)

model.train()

start = time.time()

step, loss_number = train(
  dataloader,
  model,
  optimizer,
  step,
)

end = time.time()

logger.info('Wall Time (sec)', end - start)

save_weights(
  model,
  optimizer,
  step,
  loss_number,
)

save_architecture(
  model,
)

wandb.finish()
