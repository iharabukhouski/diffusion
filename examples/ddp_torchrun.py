# #! /usr/bin/env python3

import os
import torch
# # from torch import nn
# # from torch.optim import Adam
import torch.distributed as dist
# # import torch.multiprocessing as mp
# # from torch.nn.parallel import DistributedDataParallel as DDP

# # class MyModel(nn.Module):

# #   def __init__(
# #     self,
# #   ):

# #     super().__init__()

# #     self.layer = nn.Linear(2, 2, bias = False)

# #   def forward(
# #     self,
# #     x,
# #   ):

# #     return self.layer(x)


def main(
#   rank,
):

#   pass

  print('main')

  print(torch.distributed.is_initialized())

  dist.init_process_group(
    # backend = dist.Backend.GLOO,
    backend="nccl",
    # rank = 0,
    # world_size = int(os.getenv('WORLD_SIZE')),
  )

  print(torch.distributed.is_initialized())

#   torch.distributed.monitored_barrier()

  # model = MyModel()

  # model = DDP(
  #   model,
  #   # device_ids=[
  #   #   rank,
  #   # ],
  #   # output_device = rank,
  #   # find_unused_parameters = True,
  # )

  # optimizer = Adam(
  #   model.parameters(),
  #   lr = 1,
  # )

  dist.barrier()

  dist.destroy_process_group()

  print(torch.distributed.is_initialized())


# if __name__ == '__main__':

#   os.system('clear')

#   os.environ['TORCH_CPP_LOG_LEVEL'] = 'INFO'
#   os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

#   # os.environ['RANK'] = '0'
#   # os.environ['WORLD_SIZE'] = '1'
#   # os.environ['MASTER_ADDR'] = '::1'
#   # os.environ['MASTER_PORT'] = '8080'

#   # mp.spawn(
#   #   main,
#   #   # args=(
#   #   #   config.NUMBER_OF_GPUS,
#   #   #   group_id,
#   #   # ),
#   #   args = (),
#   #   # nprocs = config.NUMBER_OF_GPUS,
#   #   nprocs = 1,
#   #   join = True, # wait for child processes to complete
#   # )

# os.environ['NCCL_DEBUG'] = 'WARN'
# os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
# os.environ['TORCH_CPP_LOG_LEVEL'] = 'INFO'
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

main()

"""

torchrun \
--nnodes=2 \
--nproc_per_node=1 \
--node_rank=0 \
--rdzv_id=1 \
--rdzv_backend=c10d \
--rdzv_endpoint=127.0.0.1:8081 \
./train.py

torchrun \
--nnodes=2 \
--nproc_per_node=1 \
--node_rank=1 \
--rdzv_id=1 \
--rdzv_backend=static \
--rdzv_endpoint=199.195.151.121:40660 \
./train.py

---

NCCL_DEBUG=WARN \
TORCH_CPP_LOG_LEVEL=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
torchrun \
--nnodes=2 \
--nproc_per_node=1 \
--node_rank=0 \
--rdzv_id=1 \
--rdzv_backend=static \
--rdzv_endpoint=127.0.0.1:8081 \
./train.py

NCCL_DEBUG=WARN \
TORCH_CPP_LOG_LEVEL=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
torchrun \
--nnodes=2 \
--nproc_per_node=1 \
--node_rank=1 \
--rdzv_id=1 \
--rdzv_backend=static \
--rdzv_endpoint=70.26.191.90:40042 \
./train.py

---

NCCL_DEBUG=WARN \
TORCH_CPP_LOG_LEVEL=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
torchrun \
--nnodes=2 \
--nproc_per_node=1 \
--node_rank=0 \
--master_addr=mymaster \
--master_port=40723 \
./train.py

NCCL_DEBUG=WARN \
TORCH_CPP_LOG_LEVEL=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
torchrun \
--nnodes=2 \
--nproc_per_node=1 \
--node_rank=1 \
--master_addr=mymaster \
--master_port=40723 \
./train.py

---

torchrun \
--rdzv_backend=static \
./train.py

---

mpirun -np 2 \
-f machines.txt \
-x MASTER_ADDR=192.168.1.66 \
-x MASTER_PORT=5201 \
-x PATH \
-bind-to none \
-map-by slot \
-mca pml ob1 \
-mca btl ^openib \
--verbose \
./train.py


"""
