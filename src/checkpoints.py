import os
import shutil
import torch
import wandb
import config
from collections import OrderedDict
import re

# DDP checkpoint to Non-DDP checkpoint
def ddp_checkpoint(
  state_dict,
):

  _state_dict = OrderedDict()
  pattern = re.compile('module.')

  for key, value in state_dict.items():

    if re.search('module', key):

        _state_dict[re.sub(pattern, '', key)] = value
    else:

        _state_dict[key] = value

  return _state_dict

class Checkpoint:

  def __init__(
    self,
    _logger,
    device,
    run_id,
    rank,
  ):

    self.logger = _logger('CHECKPOINT')

    self.disabled = not config.WANDB

    if self.disabled:

      self.logger.info('Disabled')

    else:

      self.run_id = run_id

      self.device = device

      self.rank = rank

      self.logger.debug('Init')

      self.logger.info('RUN', self.run_id)

      if not self.rank == 0:

        os.environ['WANDB_SILENT'] = 'true'

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

    if self.disabled:

      self.logger.info('Saving Weights Disabled')

    else:

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

      self.logger.info('Uploading...')

      # Includes checkpoint in wandb run
      self.run.save(
        config.CHECKPOINT_PATH,
        # policy = 'now'
      )

      self.logger.info('Uploaded')

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
    
    if not os.getenv('RUN'):
    # if not self.run_id:

      self.logger.debug('No Run ID')

      return
    
    if not config.WANDB:

      self.logger.debug('WANDB disabled')

      return

    self.logger.info('Downloading...')

    # checkpoint_filehandler = run.restore(config.CHECKPOINT_PATH)
    checkpoint_filehandler = wandb.restore(
      config.CHECKPOINT_PATH,
      # run_path: str | None = None,
      f'{config.WANDB_USERNAME}/{config.WANDB_PROJECT}/{self.run_id}_0',
      # replace: bool = False,
      # root: str | None = None
    )

    shutil.copyfile(
      checkpoint_filehandler.name,
      config.CHECKPOINT_PATH,
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
    ddp = False
  ):

    default_step = 0

    if self.disabled:

      self.logger.debug('Disabled')

      return default_step

    if not self.run.resumed:

      self.logger.info('No Checkpoint')

      return default_step

    checkpoint = torch.load(
      # checkpoint_filehandler.name,
      config.CHECKPOINT_PATH,
      map_location = self.device._device,
    )

    if ddp:

      model_state_dict = checkpoint['model_state_dict']

    else:

      model_state_dict = ddp_checkpoint(checkpoint['model_state_dict'])

    model.load_state_dict(model_state_dict)

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
    **kwargs,
  ):

    if self.disabled:

      self.logger.debug('Disabled')

    else:

      self.run.log(
        *args,
        **kwargs,
      )

  def destroy(
    self,
  ):

    if self.disabled:

      self.logger.debug('Disabled')

    else:

      self.run.finish()
