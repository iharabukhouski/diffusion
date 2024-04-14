#! /usr/bin/env python3

import os

import time
from functools import partial
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from data import create_dataloader
import config
from logger import Logger
from device import Device
from model import UNet
from scheduler import images_to_images_with_noise_at_timesteps
from checkpoints import Checkpoint
from distributed import Distributed

import wandb

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
  device,
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

    # dataloader.sampler.set_epoch(epoch)

    # images / x_0 of (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    # for batch, (images, labels) in enumerate(dataloader):
    for batch, images in enumerate(dataloader):

      # print('\n')

      # for p in model.parameters():

      #   logger.debug('1 PRM', f'{p[0][0].item():.5f}')
      #   logger.debug('1 GRD', f'{p.grad[0][0].item():.5f}' if p.grad is not None else None)

      compute_start = time.time()

      optimizer.zero_grad() # sets .grad to None

      # TODO: Can dataloader be configured in a way that the date is on the device by default?
      images = images.to(device._device)

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

def is_first_global_device():

  global_rank = int(os.getenv('RANK'))

  return global_rank == 0

def is_first_local_device():

  local_rank = int(os.getenv('LOCAL_RANK'))

  return local_rank == 0

# This code is executed `NUMBER_OF_GPUS` times
def main():

  global_rank = int(os.getenv('RANK'))
  group_rank = int(os.getenv('GROUP_RANK'))
  local_rank = int(os.getenv('LOCAL_RANK'))
  world_size = int(os.getenv('WORLD_SIZE'))

  torch.manual_seed(global_rank)

  _logger = partial(Logger, global_rank)
  logger = _logger('TRAIN')

  logger.debug('Init')
  logger.debug('PID', os.getpid())

  logger.debug('Global Rank', global_rank)
  logger.debug('Group Rank', group_rank)
  logger.debug('Local Rank', local_rank)
  logger.debug('World Size', world_size)

  device = Device(
    _logger,
    local_rank,
  )

  distributed = Distributed(
    _logger,
    device,
  )

  if is_first_global_device():

    __run_id = os.getenv('RUN', wandb.util.generate_id())

    _run_id = [__run_id] * world_size

  else:

    _run_id = [None] * world_size

  torch.distributed.broadcast_object_list(
    _run_id,
    src = 0,
  )

  print(_run_id)

  run = Checkpoint(
    _logger,
    device,
    _run_id[global_rank],
    global_rank,
  )

  if is_first_local_device():

    run.download_checkpoint()

  # distributed.barrier()

  dataloader = create_dataloader(
    _logger,
    device,
    local_rank,
    world_size,
  )

  model = UNet()

  # TODO: should be removed; needed for local cpu run
  model.to('cpu')

  model = DDP(
    model,
    # device_ids=[
    #   local_rank,
    # ],
    # output_device = local_rank,
    # find_unused_parameters = True,
  )

  # watch gradients only for rank 0
  # if is_first_local_device():

  #   run.watch(model)

  #   # run.watch(
  #   #   models = model,
  #   #   log = 'all',
  #   #   log_freq = 1,
  #   #   log_graph = True,
  #   # )

  optimizer = Adam(
    model.parameters(),
    lr = config.LEARNING_RATE,
  )

  step = run.load_weights(
    model,
    optimizer,
    ddp = True,
  )

  model.train()

  start = time.time()

  step, loss_number = train(
    logger,
    device,
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

  if is_first_local_device():

    run.save_weights(
      model,
      optimizer,
      step,
      loss_number,
    )

    # run.save_architecture(
    #   model,
    # )

  run.destroy()

  # wait for worker 0 to save checkpoints
  # distributed.barrier()

  distributed.destroy()

def is_parent_process():

  return __name__ == '__main__'

if is_parent_process():

  # os.system('clear')

  main()

"""

NCCL_DEBUG=WARN \
TORCH_CPP_LOG_LEVEL=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
RUN=nmhwij7c \
LOG=1 \
DS=128 \
BS=16 \
CPU=1 \
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--node_rank=0 \
--master_addr=localhost \
--master_port=40723 \
./train.py

"""
