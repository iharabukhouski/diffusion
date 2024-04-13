#! /usr/bin/env python3

import os
import time
import PIL
from torchvision import transforms
import safetensors
import safetensors.torch

start = time.time()

def normalize(t):

  return (t * 2) - 1

transform = transforms.Compose(
  [
    
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

dataset_path = os.path.join('../data', 'anime.safetensors')

safetensors.torch.save_file(dataset, dataset_path)

end = time.time()

print('Samples:', len(files))
print('Time:', end - start)
