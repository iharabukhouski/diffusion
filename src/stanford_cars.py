import torch
from torchvision import transforms, datasets
import config

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
      transforms.RandomHorizontalFlip(),
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
