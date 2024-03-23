import os
import torch
import logger

CPU = torch.device('cpu')
MPS = torch.device('mps')
CUDA = torch.device('cuda')

DEVICE = CPU

# DEVICE
if os.getenv('MPS', '0') == '1':

  assert torch.backends.mps.is_available() == True, 'MPS is not available'

  logger.info('[DEVICE] MPS')

  torch.set_default_device('mps')

  DEVICE = MPS

elif os.getenv('CUDA', '0') == '1':

  assert torch.cuda.is_available() == True, 'CUDA is not available'

  logger.info('[DEVICE] CUDA')

  torch.set_default_device('cuda')

  DEVICE = CUDA

else:

  logger.info('[DEVICE] CPU')

  DEVICE = CPU
