import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

class Model(nn.Module):

  def __init__(
    self,
  ):

    super().__init__()

  def forward(
    self,
    x,
  ):

    return x

def is_parent_process():

  return __name__ == '__main__'

def is_child_process():

  return __name__ == '__mp__main__'

def main(
  rank,
):

  dist.init_process_group(backend="nccl")

  model = Model()

  model = DDP(
    model,
    device_ids=[
      rank,
    ],
    output_device = rank,
    # find_unused_parameters = True,
  )

  dist.barrier()

  dist.destroy_process_group()

# This file is executed `NUMBER_OF_GPUS + 1` times
# This code is executed only once
if __name__ == '__main__':

  # os.environ['MASTER_ADDR'] = 'localhost'
  # os.environ['MASTER_ADDR'] = '127.0.0.1'
  # os.environ['MASTER_ADDR'] = '0.0.0.0'
  # os.environ['MASTER_PORT'] = '12355' # select any idle port on your machine

  NUMBER_OF_GPUS = 2

  mp.spawn(
    main,
    args=(),
    nprocs = NUMBER_OF_GPUS,
    join = True, # wait for child processes to complete
  )
