import os
import torch
from torch.utils.data import Dataset
from safetensors import safe_open
from utils import perf

class AnimeDataset(Dataset):

  def __init__(
    self,
    _logger,
    path,
    device,
  ):

    self.logger = _logger('DATASET')
    self.path = path
    self.device = device

    with perf(self.logger, 'Init'):

      self.logger.info('Anime')

      self.files = os.listdir(self.path)

      # self.file = safe_open(
      #   path,
      #   framework = 'pt',
      #   device = device,
      # )

      # self.samples = torch.load(
      #   path,
      #   map_location = device._device,
      #   # map_location = 'cpu'
      # )

  def __len__(
    self,
  ):

    # return len(self.file.keys())
    # return len(self.samples)
    return len(self.files)

  def __getitem__(
    self,
    i,
  ):

    # return self.file.get_tensor(str(i))
    # return self.samples[str(i)]
    return torch.load(
      os.path.join(self.path, str(i)),
      map_location = self.device._device,
      # map_location = 'cpu'
    )
