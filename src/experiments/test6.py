#! /usr/bin/env python3

# watch -n1 nvidia-smi

import torch

torch.set_default_device('cuda')

a = torch.rand(20000,20000)

print('Device', a.device)
print('Allocated:', round(torch.cuda.memory_allocated(0)/10243,1), 'GB')

while True:
  a += 1
  a -= 1
