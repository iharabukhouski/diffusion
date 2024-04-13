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

    with perf(self.logger, 'Init'):

      self.logger.info('Anime')

      # self.file = safe_open(
      #   path,
      #   framework = 'pt',
      #   device = device,
      # )
      self.samples = torch.load(
        path,
        map_location = device._device,
        # map_location = 'cpu'
      )

  def __len__(
    self,
  ):

    # return len(self.file.keys())
    return len(self.samples)

  def __getitem__(
    self,
    i,
  ):

    # return self.file.get_tensor(str(i))
    return self.samples[str(i)]
