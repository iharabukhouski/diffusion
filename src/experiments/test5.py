#! /usr/bin/env python3

import torch
from lovely_tensors import plot, set_config

# lt.monkey_patch()

set_config(
#   fig_close = False,
#   fig_show = True,
)

x = torch.randn((1, 10000))

a = 1

plot(x).fig.savefig(f'./run/fig{a}.png')
