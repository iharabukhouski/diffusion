import os
import torch
from functools import partial

CPU = torch.device('cpu')
MPS = torch.device('mps')
CUDA = torch.device('cuda')

def default_device(
  logger,
  rank,
):

  if torch.cuda.is_available():

    logger.info('CUDA')

    torch.set_default_device(rank)

    # return CUDA
    return torch.device(f'cuda:{rank}')

  elif torch.backends.mps.is_available():

    logger.info('MPS')

    torch.set_default_device('mps')

    return MPS

  else:

    logger.info('CPU')

    return CPU

def init(
  logger,
  rank,
):

  logger = partial(logger, 'DEVICE')()

  logger.debug('Init')

  if os.getenv('MPS', '0') == '1':

    assert torch.backends.mps.is_available() == True, 'MPS is not available'

    logger.info('MPS')

    torch.set_default_device('mps')

    return MPS

  elif os.getenv('CUDA', '0') == '1':

    assert torch.cuda.is_available() == True, 'CUDA is not available'

    logger.info('CUDA')

    torch.set_default_device(rank)

    return torch.device(f'cuda:{rank}')
    # return rank

  elif os.getenv('CPU', '0') == '1':

    logger.info('CPU')

    return CPU

  else:

    return default_device(
      logger,
      rank,
    )
