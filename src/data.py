import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
import numpy as np
import config
import logger
import device

def create_dataset():

  """
  Returns a data set with transformations applied
  """

  transform = transforms.Compose(
    [
      transforms.Resize(
        (
          config.IMG_SIZE,
          config.IMG_SIZE,
        ),
      ),
      # transforms.RandomHorizontalFlip(),
      transforms.ToTensor(), # scale data to [0, 1]
      transforms.Lambda(lambda t: (t * 2) - 1) # scale data to [-1, 1]
    ]
  )

  train = datasets.StanfordCars(
    root = '../data',
    download = False,
    transform = transform,
  )

  test = datasets.StanfordCars(
    root = '../data',
    download = False,
    transform = transform,
    split = 'test',
  )

  return torch.utils.data.ConcatDataset(
    [
      train,
      test,
    ],
  )

def create_dataloader():

  dataset = create_dataset()

  _indices = list(range(len(dataset)))
  indices = _indices if config.DATASET_SIZE is None else _indices[:config.DATASET_SIZE]

  dataloader = DataLoader(
    dataset,
    batch_size = config.BATCH_SIZE,
    # shuffle = True,
    drop_last = True,

    # TODO: If I understand correctly pinned memory on CUDA is supported only for dense tensors
    pin_memory = False if device.DEVICE == device.CUDA else True,
    sampler = SubsetRandomSampler(indices),

    # NOTE: Due to PyTorch crash on MPS
    generator = torch.Generator(device=device.DEVICE),
  )

  return dataloader

def print_dataloader(
  dataloader,
):

  number_of_batches = len(dataloader)
  batch_size = dataloader.batch_size

  logger.info('[DATA] Dataset Size:', number_of_batches * batch_size)
  logger.info('[DATA] Number of Batches:', number_of_batches)

def x_to_PIL(
  x, # a.k.a. image
):

  reverse_transform = transforms.Compose(
    [
      transforms.Lambda(lambda t: (t + 1) / 2), # scale data to [0, 1]
      
      # Channel,Height,Width -> Height,Width,Channel
      transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC

      transforms.Lambda(lambda t: t * 255.), # scale data to [0, 255]
      transforms.Lambda(lambda t: t.to(device.CPU).numpy().astype(np.uint8)), # convert from PyTorch tensors to Numpy Array
      transforms.ToPILImage(),
    ]
  )

  return reverse_transform(x)
