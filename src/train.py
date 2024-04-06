#! /usr/bin/env python3

import os

import time
import torch
from torch.optim import Adam
import torch.nn.functional as F
from data import create_dataloader
import config
from logger import Logger
import device as Device
import wandb
from model import UNet
from scheduler import images_to_images_with_noise_at_timesteps
import checkpoints
from distributed import Distributed

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

      if config.WANDB:

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


from functools import partial

# This code is executed `NUMBER_OF_GPUS` times
def main():

  global_rank = os.getenv('RANK')
  group_rank = os.getenv('GROUP_RANK')
  local_rank = os.getenv('LOCAL_RANK')
  world_size = os.getenv('WORLD_SIZE')

  # TODO: We need to have the same RUN for all processes
  group_id = os.getenv('RUN') # wandb.util.generate_id()

  torch.manual_seed(global_rank)

  _logger = partial(Logger, global_rank)
  logger = _logger('TRAIN')

  logger.debug('Init')
  logger.debug('PID', os.getpid())

  logger.debug('Global Rank', global_rank)
  logger.debug('Group Rank', group_rank)
  logger.debug('Local Rank', local_rank)
  logger.debug('World Size', world_size)

  device = Device.init(
    _logger,
    local_rank,
  )

  run = checkpoints.init(
    _logger,
    group_id,
    global_rank,
  )

  distributed = Distributed(
    _logger,
    device,
  )

  # if local_rank == 0 and os.getenv('RUN') and config.WANDB:

  #   checkpoints.download_checkpoint(
  #     logger,
  #     group_id,
  #   )

  # dist.barrier()

  # dataloader = create_dataloader(
  #   _logger,
  #   local_rank,
  #   world_size,
  # )

  # model = UNet()

  # model = DDP(
  #   model,
  #   # device_ids=[
  #   #   rank,
  #   # ],
  #   # output_device = rank,
  #   # find_unused_parameters = True,
  # )

  # # watch gradients only for rank 0
  # # if rank == 0:

  # #   run.watch(model)

  #   # run.watch(
  #   #   models = model,
  #   #   log = 'all',
  #   #   log_freq = 1,
  #   #   log_graph = True,
  #   # )

  # optimizer = Adam(
  #   model.parameters(),
  #   lr = config.LEARNING_RATE,
  # )

  # step = checkpoints.load_weights(
  #   logger,
  #   device,
  #   run,
  #   model,
  #   optimizer,
  # )

  # model.train()

  # start = time.time()

  # step, loss_number = train(
  #   logger,
  #   device,
  #   run,
  #   dataloader,
  #   model,
  #   optimizer,
  #   step,
  # )

  # end = time.time()

  # logger.info(f'Step: {step}')
  # logger.info(f'Loss: {loss_number:.4f}')
  # logger.info(f'Time: {end - start:.4f}s')

  # if local_rank == 0:

  #   checkpoints.save_weights(
  #     logger,
  #     run,
  #     model,
  #     optimizer,
  #     step,
  #     loss_number,
  #   )

  #   # save_architecture(
  #   #   run,
  #   #   model,
  #   # )

  # # TODO: we need a class
  # if config.WANDB:

  #   run.finish()

  # wait for worker 0 to save checkpoints
  # dist.barrier()

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
LOG=1 \
WANDB=0 \
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--node_rank=0 \
--master_addr=localhost \
--master_port=40723 \
./train.py

"""
