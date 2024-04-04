import os
import torch
import wandb
import config
import logger
from functools import partial
from collections import OrderedDict
import re

def init(
  _logger,
  group,
  rank,
):

  logger = partial(_logger, 'CHECKPOINT')()

  logger.debug('Init')

  # os.environ['WANDB_SILENT'] = 'true'

  wandb.login(
    key = config.WANDB_API_KEY,
  )

  run = wandb.init(

    project = config.WANDB_PROJECT,

    # group = config.WANDB_GROUP,
    group = group,

    # SEE: https://docs.wandb.ai/ref/python/init
    # SEE: https://docs.wandb.ai/guides/runs/resuming
    # id = config.WANDB_RUN_ID,
    id = f'{group}_{rank}',
    # resume = config.WANDB_RESUME,
    resume = 'allow',
    # name = config.WANDB_NAME,
    name = f'{group}_{rank}',

    config = {
      # Architecture
      'T': config.T,
      'IMG_SIZE': config.IMG_SIZE,
      # Training
      'LEARNING_RATE': config.LEARNING_RATE,
      'NUMBER_OF_EPOCHS': config.NUMBER_OF_EPOCHS,
      'BATCH_SIZE': config.BATCH_SIZE,
      'DATASET_SIZE': config.DATASET_SIZE,
    },
  )

  return run

def save_weights(
  run,
  model,
  optimizer,
  step,
  loss,
):

  logger.info('[CHECKPOINT] Save Weights: Step', step)
  logger.info('[CHECKPOINT] Save Weights: Loss', loss)

  # Create checkpoint
  torch.save(
    {
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'step': step,
      'loss': loss,
    },
    config.CHECKPOINT_PATH,
  )

  # Includes checkpoint in wandb run
  run.save(
    config.CHECKPOINT_PATH,
    # policy = 'now'
  )

def save_architecture(
  run,
  model,
):

  # Generating fake input because torch requires it
  images_with_noise_at_timesteps = torch.randn(1, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)
  timesteps = torch.randn(1)

  # Exporting model architecture in ONNX format
  torch.onnx.export(
    model,
    (
      images_with_noise_at_timesteps,
      timesteps,
    ),
    config.MODEL_ONNX_PATH,
  )

  # NOTE: Does not work because the latest Python is not supported
  # onnx_program = torch.onnx.dynamo_export(model, x)

  # Adding ONNX file to wandb
  run.save(
    config.MODEL_ONNX_PATH,
  )

import torch.distributed as dist

def download_checkpoint(
  logger,
  group,
):

  logger.info('Downloading...')

  # checkpoint_filehandler = run.restore(config.CHECKPOINT_PATH)
  wandb.restore(
    config.CHECKPOINT_PATH,
    # run_path: str | None = None,
    f'{config.WANDB_USERNAME}/{config.WANDB_PROJECT}/{group}_0',
    # replace: bool = False,
    # root: str | None = None
  )

  # logger.info('Path', checkpoint_filehandler.name)

  # checkpoint_filehandler.close()

  # TODO: wandb.save does not replace 'checkpoint.tar' if it already exists
  # os.remove(checkpoint_filehandler.name)

  logger.info('Downloaded')

def load_weights(
  logger,
  _device,
  run,
  model,
  optimizer = None,
):

  if run.resumed:

    checkpoint = torch.load(
      # checkpoint_filehandler.name,
      config.CHECKPOINT_PATH,
      map_location = _device,
    )

    model.load_state_dict(ddp_checkpoint(checkpoint['model_state_dict']))

    logger.info('Model Restored')

    if optimizer is not None:

      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

      logger.info('Optimizer Restored')

    logger.info('Step', checkpoint['step'])
    logger.info('Loss', checkpoint['loss'])

    return checkpoint['step']

  else:

    logger.info('No Checkpoint')

    step = 0

    logger.info('Step', step)

    return step

# DDP checkpoint to Non-DDP checkpoint
def ddp_checkpoint(
  state_dict,
):

  _state_dict = OrderedDict()
  pattern = re.compile('module.')

  for k, v in state_dict.items():

    if re.search('module', k):

        _state_dict[re.sub(pattern, '', k)] = v
    else:

        _state_dict[k] = v

  return _state_dict
