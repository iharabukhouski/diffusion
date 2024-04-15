import os
import torch

CPU = torch.device('cpu')
MPS = torch.device('mps')

def default_device(
  logger,
  rank,
):

  if torch.cuda.is_available():

    logger.info('CUDA')

    torch.set_default_device(rank)

    # Needed for Torch Elastic on CUDA
    torch.cuda.set_device(rank)

    return torch.device(f'cuda:{rank}')

  elif torch.backends.mps.is_available():

    logger.info('MPS')

    torch.set_default_device('mps')

    return MPS

  else:

    logger.info('CPU')

    return CPU

class Device:

  def __init__(
    self,
    logger,
    rank,
  ):

    self.logger = logger('DEVICE')

    self.logger.debug('Init')

    if os.getenv('MPS', '0') == '1':

      assert torch.backends.mps.is_available() == True, 'MPS is not available'

      self.logger.info('MPS')

      torch.set_default_device('mps')

      self._device = MPS

    elif os.getenv('CUDA', '0') == '1':

      assert torch.cuda.is_available() == True, 'CUDA is not available'

      self.logger.info('CUDA')

      # Needed for Torch Elastic on CUDA
      torch.cuda.set_device(rank)

      torch.set_default_device(rank)

      self._device = torch.device(f'cuda:{rank}')

    elif os.getenv('CPU', '0') == '1':

      self.logger.info('CPU')

      self._device = CPU

    else:

      self._device = default_device(
        self.logger,
        rank,
      )

  def is_cuda(
    self,
  ):

    return self._device.type.startswith('cuda')
