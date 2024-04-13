import os
import torch
import wandb
import config
import logger
from collections import OrderedDict
import re

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

class Checkpoint:

  def __init__(
    self,
    _logger,
    device,
    rank,
  ):

    self.logger = _logger('CHECKPOINT')

    if not config.WANDB:

      logger.info('WANDB Disabled')

    # TODO: We need to have the same RUN for all processes
    self.run_id = os.getenv('RUN') # wandb.util.generate_id()

    self.device = device

    logger.debug('Init')

    # os.environ['WANDB_SILENT'] = 'true'

    wandb.login(
      key = config.WANDB_API_KEY,
    )

    self.run = wandb.init(

      project = config.WANDB_PROJECT,

      # group = config.WANDB_GROUP,
      group = self.run_id,

      # SEE: https://docs.wandb.ai/ref/python/init
      # SEE: https://docs.wandb.ai/guides/runs/resuming
      # id = config.WANDB_RUN_ID,
      id = f'{self.run_id}_{rank}',
      # resume = config.WANDB_RESUME,
      resume = 'allow',
      # name = config.WANDB_NAME,
      name = f'{self.run_id}_{rank}',

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

  def save_weights(
    self,
    model,
    optimizer,
    step,
    loss,
  ):

    if config.WANDB:

      self.logger.info('Save Weights: Step', step)
      self.logger.info('Save Weights: Loss', loss)

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
      self.run.save(
        config.CHECKPOINT_PATH,
        # policy = 'now'
      )

    else:

      self.logger.info('Saving Weights Disabled')

  def save_architecture(
    self,
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
    self.run.save(
      config.MODEL_ONNX_PATH,
    )

  def download_checkpoint(
    self,
  ):
    
    if not self.run_id:

      self.logger.debug('No Run ID')

      return
    
    if not config.WANDB:

      self.logger.debug('WANDB disabled')

      return

      self.logger.info('Downloading...')

    # checkpoint_filehandler = run.restore(config.CHECKPOINT_PATH)
    wandb.restore(
      config.CHECKPOINT_PATH,
      # run_path: str | None = None,
      f'{config.WANDB_USERNAME}/{config.WANDB_PROJECT}/{self.run_id}_0',
      # replace: bool = False,
      # root: str | None = None
    )

    # self.logger.info('Path', checkpoint_filehandler.name)

    # checkpoint_filehandler.close()

    # TODO: wandb.save does not replace 'checkpoint.tar' if it already exists
    # os.remove(checkpoint_filehandler.name)

    self.logger.info('Downloaded')

  def load_weights(
    self,
    model,
    optimizer = None,
  ):
    
    default_step = 0

    if not config.WANDB:

      logger.debug('WANDB disabled')

      return default_step
    
    if not self.run.resumed:

      self.logger.info('No Checkpoint')

      return default_step

    checkpoint = torch.load(
      # checkpoint_filehandler.name,
      config.CHECKPOINT_PATH,
      map_location = self._device,
    )

    model.load_state_dict(ddp_checkpoint(checkpoint['model_state_dict']))

    self.logger.info('Model Restored')

    if optimizer is not None:

      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

      self.logger.info('Optimizer Restored')

    self.logger.info('Step', checkpoint['step'])
    self.logger.info('Loss', checkpoint['loss'])

    return checkpoint['step']

  def log(
    self,
    *args,
  ):

    if not config.WANDB:

      self.logger.debug('WANDB disabled')

    else:

      self.run.log(*args)

  def destroy(
    self,
  ):

    if not config.WANDB:

      self.logger.debug('WANDB disabled')

    else:

      self.run.finish()
