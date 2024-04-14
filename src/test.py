#! /usr/bin/env python3

import torch

a = torch.load(
  '../data/anime_processed/0',
  map_location = 'cpu',
)

print(a)
print(a.shape)
print(a.min())
print(a.max())
