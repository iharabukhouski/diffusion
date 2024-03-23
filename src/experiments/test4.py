#! /usr/bin/env python3

# track experiments
# visualize results
# spot regressions

import wandb
import random

wandb.login(
  key = '5a6073436af3e3fef9b97d298ab52b35f9d3df40'
)

wandb.init(
  project = 'Test',
  config = {
    'learning_rate': 0.02,
    'architecturn': 'CNN',
    'dataset': 'CIFAR-100',
    'epochs': 10,
  },
)

epochs = 10
offset = random.random() / 5

for epoch in range(2, epochs):
  
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset
  loss = 2 ** -epoch + random.random() / epoch + offset

  wandb.log(
    {
      'acc': acc,
      'loss': loss,
    }
  )

wandb.finish()
