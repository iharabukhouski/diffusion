#! /usr/bin/env python3

import os
import sys

# os.system('clear')

import time
import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F
from data import create_dataloader
import config
# import logger
from logger import Logger
import device
import wandb
from model import UNet, UNet2, print_model
from scheduler import images_to_images_with_noise_at_timesteps
import checkpoints

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

# from torchtnt.utils.flops import FlopTensorDispatchMode
# from flop_counter import FlopCounterMode

def train(
  logger,
  _device,
  run,
  dataloader,
  model,
  optimizer,
  step,
):

  # logger.info('PARAM', next(model.parameters())[0][0])

  # flop_counter = FlopCounterMode(model)

  # with flop_counter:

  # for p in model.parameters():

  #   logger.info('INIT PRM', f'{p[0][0].item():.5f}')
  #   logger.info('INIT GRD', f'{p.grad[0][0].item():.5f}' if p.grad is not None else None)

  for epoch in range(config.NUMBER_OF_EPOCHS):

    step_start = time.time()

    dataloader.sampler.set_epoch(epoch)

    # images / x_0 of (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    for batch, (images, labels) in enumerate(dataloader):

      # print('\n')



      # for p in model.parameters():

      #   logger.debug('1 PRM', f'{p[0][0].item():.5f}')
      #   logger.debug('1 GRD', f'{p.grad[0][0].item():.5f}' if p.grad is not None else None)

      compute_start = time.time()

      optimizer.zero_grad() # sets .grad to None

      # TODO: Can dataloader be configured in a way that the date is on the device by default?
      images = images.to(_device)

      # timesteps of (BATCH_SIZE)
      timesteps = torch.randint(
        0, # low
        config.T, # high
        (
            config.BATCH_SIZE,
        ), # size
        dtype = torch.int64,
      )

      # with FlopTensorDispatchMode(model) as ftdm:

      loss = calculate_loss(
        model,
        images,
        timesteps,
      )
      loss_number = loss.item()

        # print('FLOPS', ftdm.flop_counts)

        # ftdm.reset()

      # for p in model.parameters():

      #   logger.debug('2 PRM', f'{p[0][0].item():.5f}')
      #   logger.debug('2 GRD', f'{p.grad[0][0].item():.5f}' if p.grad is not None else None)

      loss.backward() # calculates .grad

      # for p in model.parameters():

      #   logger.debug('3 PRM', f'{p[0][0].item():.5f}')
      #   logger.debug('3 GRD', f'{p.grad[0][0].item():.5f}' if p.grad is not None else None)


      optimizer.step() # updates params

      # for p in model.parameters():

      #   logger.debug('4 PRM', f'{p[0][0].item():.5f}')
      #   logger.debug('4 GRD', f'{p.grad[0][0].item():.5f}' if p.grad is not None else None)

      run.log(
        {
          'loss': loss_number,
        },
        step = step,
      )

      compute_end = time.time()

      step_end = time.time()

      if step % config.LOG_EVERY == 0:

        logger.info(f'Step: {step:>4} | Epoch: {epoch:>4} | Batch: {batch:>4} | Loss: {loss_number:.4f} | Compute: {compute_end - compute_start:.4f}s | Step: {step_end - step_start:.4f}s')

      step_start = time.time()

      step += 1

  # for p in model.parameters():

  #   logger.info('FIN PRM', f'{p[0][0].item():.5f}')
  #   logger.info('FIN GRD', f'{p.grad[0][0].item():.5f}' if p.grad is not None else None)

  return step, loss_number

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(
  rank,
  world_size,
):

  # TODO: I need to compile torch from source on the machine that has MPI installed
  dist.init_process_group(
    # backend = dist.Backend.MPI,
    rank = rank,
    world_size = world_size,
  )

def cleanup():

  dist.destroy_process_group()

from functools import partial

# This code is executed `NUMBER_OF_GPUS` times
def main(
  rank,
  world_size,
  wandb_group,
):

  torch.manual_seed(rank)

  _logger = partial(Logger, rank)
  logger = partial(_logger, 'TRAIN')()

  logger.debug('Init')

  logger.info('PID', os.getpid())

  _device = device.init(
    _logger,
    rank,
  )

  # print('Process:', __name__)

  # print('Torch Version:', torch.__version__)

  # NUM_OF_CPU = os.cpu_count()

  # print('CPU Count:', NUM_OF_CPU)

  run = checkpoints.init(
    _logger,
    wandb_group,
    rank,
  )
  # run = None

  setup(
    rank,
    world_size,
  )

  dataloader = create_dataloader(
    _logger,
    rank,
    world_size,
  )

  model = UNet()
  # model = UNet2()

  # logger.info('MODEL DEVICE', model)

  model = DDP(
    model,
    device_ids=[
      rank,
    ],
    output_device = rank,
    # find_unused_parameters = True,
  )

  # watch gradients only for rank 0
  # if rank == 0:

  #   run.watch(model)

    # run.watch(
    #   models = model,
    #   log = 'all',
    #   log_freq = 1,
    #   log_graph = True,
    # )

  # print_model(model)

  optimizer = Adam(
  # optimizer = SGD(
    model.parameters(),
    lr = config.LEARNING_RATE,
  )

  step = checkpoints.load_weights(
    logger,
    _device,
    run,
    model,
    optimizer,
  )
  # step = 0

  # dist.barrier()

  model.train()

  start = time.time()

  step, loss_number = train(
    logger,
    _device,
    run,
    dataloader,
    model,
    optimizer,
    step,
  )

  end = time.time()

  logger.info(f'Step: {step}')
  logger.info(f'Loss: {loss_number:.4f}')
  logger.info(f'Time: {end - start:.4f}s')

  if rank == 0:

    # pass

    checkpoints.save_weights(
      run,
      model,
      optimizer,
      step,
      loss_number,
    )

    # save_architecture(
    #   run,
    #   model,
    # )

  run.finish()

  cleanup()


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355' # select any idle port on your machine

# This file is executed `NUMBER_OF_GPUS + 1` times

# This code is executed only once
if __name__ == '__main__':

  os.system('clear')

  _logger = partial(Logger, -1)
  logger = partial(_logger, 'MAIN')()

  group_id = os.getenv('RUN', wandb.util.generate_id())

  logger.info('PID', os.getpid())
  logger.info('RUN', group_id)

  if os.getenv('RUN'):

    checkpoints.download_checkpoint(
      logger,
      group_id,
    )

  mp.spawn(
    main,
    args=(
      config.NUMBER_OF_GPUS,
      group_id,
    ),
    nprocs = config.NUMBER_OF_GPUS,
    join = True, # wait for child processes to complete
  )
