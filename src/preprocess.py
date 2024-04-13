#! /usr/bin/env python3

import os
import time
import PIL
import torch
from torchvision import transforms
import safetensors
import safetensors.torch
import config

start = time.time()

print('Preprocessing...')

def normalize(t):

  return (t * 2) - 1

transform = transforms.Compose(
  [
    transforms.Resize(
      (
        config.IMG_SIZE,
        config.IMG_SIZE,
      ),
    ),
    transforms.ToTensor(),
    transforms.Lambda(normalize),
  ]
)

dir = '../data/anime'
files = os.listdir(dir)

dataset = {}

for i, file in enumerate(files):

  file_path = os.path.join(dir, file)

  image = PIL.Image.open(file_path)

  dataset[str(i)] = transform(image)

# safetensors.torch.save_file(
#   dataset,
#   config.ANIME_DATASET_PATH,
# )
torch.save(
  dataset,
  config.ANIME_DATASET_PATH,
)

end = time.time()

print('Preprocessing Done')
print('Samples:', len(files))
print('Time:', end - start)
