import os
from dotenv import load_dotenv
import wandb
import torch

load_dotenv()

# CONFIG

WANDB = os.getenv('WANDB', '1') == '1'
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

if WANDB:

  assert WANDB_API_KEY is not None, '"WANDB_API_KEY" is required'

WANDB_USERNAME = 'iharabukhouski'
WANDB_PROJECT = 'Test'
# WANDB_GROUP = 'MyGroup'
WANDB_RUN_ID = os.getenv('RUN', wandb.util.generate_id()) 
WANDB_RESUME = 'allow' if WANDB_RUN_ID is not None else None
# WANDB_RESUME = 'must' if WANDB_RUN_ID is not None else None
WANDB_NAME = WANDB_RUN_ID

CHECKPOINT_PATH = 'checkpoint.tar'
MODEL_ONNX_PATH = 'model.onnx'

def default_number_of_gpus():

  if torch.cuda.is_available():

    return torch.cuda.device_count()

  else:

    return 1

DEFAULT_NUMBER_OF_GPUS = default_number_of_gpus()
NUMBER_OF_GPUS = int(os.getenv('GPUS') or DEFAULT_NUMBER_OF_GPUS)

DEFAULT_NUMBER_OF_CPUS = os.cpu_count()
NUMBER_OF_CPUS = int(os.getenv('CPUS') or DEFAULT_NUMBER_OF_CPUS)

## Model
T = 100 # number of steps
TIMESTEP_EMBEDDING_DIMENTIONALITY = 32 # dimentionality of positional encodding of timesteps
# IMG_SIZE = 64
IMG_SIZE = 8
IMG_CHANNELS = 3
# CHANNELS = (IMG_CHANNELS, 64, 128, 256, 512, 1024) # depth? / using filters & convolution
CHANNELS = (IMG_CHANNELS, IMG_SIZE, 128) # depth? / using filters & convolution

## Training
DEFAULT_DATASET_SIZE = None
DATASET_SIZE = int(os.getenv('DS', '0')) or DEFAULT_DATASET_SIZE
# DATASET_SIZE = 1024
# DATASET_SIZE = None

DEFAULT_BATCH_SIZE = 128
BATCH_SIZE = int(os.getenv('BS') or DEFAULT_BATCH_SIZE)

# NUMBER_OF_EPOCHS = 100
DEFAULT_NUMBER_OF_EPOCHS = 1
NUMBER_OF_EPOCHS = int(os.getenv('EPOCHS') or DEFAULT_NUMBER_OF_EPOCHS)
LEARNING_RATE = 0.001 * NUMBER_OF_GPUS
# LEARNING_RATE = 1

LOG_EVERY = 5

# Data
MAX_PROCESSES_PER_GPU = 4
