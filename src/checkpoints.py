import os
import torch
import wandb
import config
import logger
import device

wandb.login(
  key = config.WANDB_API_KEY,
)

wandb.init(

  project = config.WANDB_PROJECT,

  # SEE: https://docs.wandb.ai/ref/python/init
  # SEE: https://docs.wandb.ai/guides/runs/resuming
  id = config.WANDB_RUN_ID,
  resume = config.WANDB_RESUME,
  name = config.WANDB_NAME,

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
  wandb.save(
    config.CHECKPOINT_PATH,
    # policy = 'now'
  )

def save_architecture(
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
  wandb.save(
    config.MODEL_ONNX_PATH,
  )

def restore_weights(
  model,
  optimizer = None,
):

  if wandb.run.resumed:

    logger.info('[CHECKPOINT] Downloading...')

    checkpoint_filehandler = wandb.restore(config.CHECKPOINT_PATH)

    logger.info('[CHECKPOINT] Downloaded')

    logger.info('[CHECKPOINT] Path', checkpoint_filehandler.name)

    checkpoint = torch.load(
      checkpoint_filehandler.name,
      map_location = device.DEVICE,
    )
    # checkpoint = torch.load('checkpoint.tar')

    checkpoint_filehandler.close()

    # TODO: wandb.save does not replace 'checkpoint.tar' if it already exists
    os.remove(checkpoint_filehandler.name)

    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info('[CHECKPOINT] Model Restored')

    if optimizer is not None:

      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

      logger.info('[CHECKPOINT] Optimizer Restored')

    logger.info('[CHECKPOINT] Step', checkpoint['step'])
    logger.info('[CHECKPOINT] Loss', checkpoint['loss'])

    return checkpoint['step']

  else:

    logger.info('[CHECKPOINT] No Checkpoint')

    step = 0

    logger.info('[CHECKPOINT] Step', step)

    return step
