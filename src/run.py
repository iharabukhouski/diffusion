#! /usr/bin/env python3

import os
from functools import partial
from logger import Logger
from device import Device
from checkpoints import Checkpoint, MODE
from model import UNet, print_model
from plt import plt_images
from sampling import sample_image

def main():

  rank = 0

  _logger = partial(Logger, rank)

  device = Device(
    _logger,
    rank,
  )

  model = UNet()

  run_id = os.getenv('RUN')

  run = Checkpoint(
    _logger,
    device,
    run_id,
    rank,
    MODE.EVAL,
  )

  run.download_checkpoint()

  run.load_weights(
    model,
    ddp = False,
  )

  # TODO: `eval` - causes issues with BatchNorm during inference
  # it is related to `track_runing_stats`
  # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/78
  # model.eval()

  images = sample_image(model)

  plt_images(images)

  # print_model(model)

if __name__ == '__main__':

  os.system('clear')

  main()

"""

RUN=sfddjolb \
./run.py

"""
