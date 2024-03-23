#! /usr/bin/env python3

import os
os.system('clear')

import torch
# import torch.nn as nn
import device
import profiler
# import package1

print('[INFO] Torch Version:', torch.__version__)

# Reproducibility
torch.manual_seed(42)


def create_betas(
  number_of_steps,
  start = 0.0001,
  end = 0.02,
):

  # Evenly spaced values from `start` to `end`
  return torch.linspace(
    start,
    end,
    number_of_steps,
  )

def main():

  # torch.ones((1))

  SIZE = 2 ** 14

  a = torch.randn((SIZE, SIZE))
  b = torch.randn((SIZE, SIZE))
  c = a @ b

  # a = torch.tensor([1])
  # a = torch.Tensor([1])

  # print(a)

  # betas = create_betas(
  #   number_of_steps = 100,
  # )
  # # alphas = 1.0 - betas

  # pass

profiler.profile(
  main,
)

# main()
