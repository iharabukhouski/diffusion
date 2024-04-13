from torch.utils.data import Dataset
import safetensors

class AnimeDataset(Dataset):

  def __init__(
    self,
    dataset_path = '../data/anime.safetensors',
    device = 'cpu',
  ):

    self.file = safetensors.safe_open(
      dataset_path,
      framework = 'pt',
      device = device,
    )

  def __len__(
    self,
  ):

    return len(self.file.keys())

  def __getitem__(
    self,
    i,
  ):

    return self.file.get_tensor(str(i))
