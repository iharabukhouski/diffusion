import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import transforms, datasets
import numpy as np
import config
import logger
import device
from torch.utils.data.distributed import DistributedSampler
from functools import partial

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def normalize(t):

  return (t * 2) - 1

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

      # TODO: I am not sure why but this causes a crash "AttributeError: Can't pickle local object 'create_dataset.<locals>.<lambda>'"
      # transforms.Lambda(lambda t: (t * 2) - 1) # scale data to [-1, 1] 
      transforms.Lambda(normalize)
    ]
  )

  train = datasets.StanfordCars(
    root = '../data',
    download = False,
    transform = transform,
  )

  # test = datasets.StanfordCars(
  #   root = '../data',
  #   download = False,
  #   transform = transform,
  #   split = 'test',
  # )

  return torch.utils.data.ConcatDataset(
    [
      train,
      # test,
    ],
  )

def create_dataloader(
  _logger,
  rank,
  world_size,
):

  logger = partial(_logger, 'DATA')()

  logger.debug('Init')

  dataset = create_dataset()

  _indices = list(range(len(dataset)))
  indices = _indices if config.DATASET_SIZE is None else _indices[:config.DATASET_SIZE]

  subset = Subset(dataset, indices)

  sampler = DistributedSampler(
    # dataset,
    subset,
    num_replicas = world_size,
    rank = rank,
    shuffle = False,
    # shuffle = True, # TODO: fails on MPS with "RuntimeError: Expected a 'mps:0' generator device but found 'cpu'"

    

    # drop_last = False,
    drop_last = True,
  )

  # SEE: https://github.com/huggingface/pytorch-image-models/pull/140/files
  # dataloader = DataLoader(
  dataloader = MultiEpochsDataLoader(
    # dataset,
    subset,
    batch_size = config.BATCH_SIZE,
    # shuffle = True,
    drop_last = True,
    # drop_last = False,

    # TODO: If I understand correctly pinned memory on CUDA is supported only for dense tensors
    # pin_memory = False if device.DEVICE == device.CUDA else True,
    pin_memory = False, # NOTE: Due to DDP

    # sampler = SubsetRandomSampler(indices),
    sampler = sampler,

    # NOTE: Due to PyTorch crash on MPS
    # generator = torch.Generator(device = device.DEVICE),
    # generator = torch.Generator(device = torch.device('mps')),

    # TODO: Should something like NUM_OF_CPUS // NUM_OF_GPUS and round
    # num_workers = config.NUM_OF_CPU
    num_workers = 2,
    # num_workers = 1,
    # num_workers = 0,

    # NOTE: Because of "RuntimeError: _share_filename_: only available on CPU"
    # SEE: https://github.com/pytorch/pytorch/issues/87688#issuecomment-1968901877
    # multiprocessing_context='fork' if device.DEVICE == device.MPS else 'spawn' if device.DEVICE == device.CUDA else None
    multiprocessing_context='spawn', #TODO: should be removed

    persistent_workers=True
  )

  # print_dataloader(
  #   logger,
  #   dataloader,
  # )

  return dataloader

# def print_dataloader(
#   logger,
#   dataloader,
# ):

#   number_of_batches = len(dataloader)
#   batch_size = dataloader.batch_size

#   logger.info('Samples:', number_of_batches * batch_size)
#   logger.info('Batches:', number_of_batches)

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
