import os
from dotenv import load_dotenv
import wandb

load_dotenv()

# CONFIG

WANDB_API_KEY = os.getenv('WANDB_API_KEY')

assert WANDB_API_KEY is not None, '"WANDB_API_KEY" is required'

WANDB_PROJECT = 'Test'
WANDB_RUN_ID = os.getenv('RUN', wandb.util.generate_id()) 
WANDB_RESUME = 'allow' if WANDB_RUN_ID is not None else None
# WANDB_RESUME = 'must' if WANDB_RUN_ID is not None else None
WANDB_NAME = WANDB_RUN_ID

CHECKPOINT_PATH = 'checkpoint.tar'
MODEL_ONNX_PATH = 'model.onnx'

## Model
T = 100 # number of steps
TIMESTEP_EMBEDDING_DIMENTIONALITY = 32 # dimentionality of positional encodding of timesteps
IMG_SIZE = 64
IMG_CHANNELS = 3
CHANNELS = (IMG_CHANNELS, 64, 128, 256, 512, 1024) # depth? / using filters & convolution
# CHANNELS = (IMG_CHANNELS, 64, 128) # depth? / using filters & convolution

## Training
DEFAULT_DATASET_SIZE = None
DATASET_SIZE = int(os.getenv('DS', '0')) or DEFAULT_DATASET_SIZE
# DATASET_SIZE = 1024

DEFAULT_BATCH_SIZE = 128
BATCH_SIZE = int(os.getenv('BS') or DEFAULT_BATCH_SIZE)

# NUMBER_OF_EPOCHS = 100
NUMBER_OF_EPOCHS = 1
LEARNING_RATE = 0.001

# PLATFORM

import torch

print('Torch Version:', torch.__version__)

torch.manual_seed(42)
