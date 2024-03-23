#! /usr/bin/env python3

import os
import torch
import torch.nn as nn
import profiler
import package1

os.system('clear')

print('[INFO] Torch Version:', torch.__version__)

# DEVICE
if os.getenv('MPS', '0') == '1':

  print('[INFO] Device: MPS')

  torch.set_default_device('mps')

else:

  print('[INFO] Device: CPU')

# Reproducibility
torch.manual_seed(42)

CPU = torch.device('cpu')
MPS = torch.device('mps')


SIZE = 2 ** 14

class MyModel(nn.Module):

  def __init__(
    self,
    in_features,
    out_features,
  ):

    super().__init__()

    self.hidden1 = nn.Linear(
      in_features,
      out_features,
      bias = False
    )

  def forward(
    self,
    x,
  ):

    return self.hidden1(x)
    # return x

def main():

  # package1.foo1()

  # a = torch.randn((SIZE, SIZE))

  # b = torch.randn((SIZE, SIZE))

  # c = a @ b

  # print(c)

  # model = MyModel(
  #   in_features = 2,
  #   out_features = 2,
  # )

  x = torch.randn((1, 2))

  # print('X', x)

  # y = model(x)

  # print(model)

  # print('W', model.state_dict())

  # print('Y', y)

  # torch.save(model.state_dict(), './weights')

  pass

profiler.profile(
  main,
)
