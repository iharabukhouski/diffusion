- data parallel
- model parallel / pipeline parallel
- tensor parallel (e.g. shard a matmul and execute on multiple machines)
- moe

- dp - i do not think one should ever use it
- ddp - data parallel; multi-gpu
- rcp
- fsdp
- torchrun - multi-node
- tp - tensor parallel

- multi-node pipeline parallel with rpc
  - https://pytorch.org/tutorials/intermediate/dist_pipeline_parallel_tutorial.html

- slurm - jobs queue; jobs scheduler; resources allocation per user; accounting

- mac + pi + static backend -> hangs

# distributed backends
- nccl - nvidia gpus
- gloo - cpus
- mpi - not recommened for some reason

# torchrun backends
- static
- c10d
- etcd - will be deprecated

# torchrun

```bash
# this segfaults on macos
torchrun \
--rdzv_backend=c10d \
./empty_file.py

# this this works on macos
torchrun \
--rdzv_backend=static \
./empty_file.py
```

# using --rdzv_id, --rdzv_backend, --rdzv_endpoint

- fails either with cannot resolve IPv4 or with cannot resolve IPv6
- tested on lambdalabs, vastai

# configuring /etc/hosts

## master

```
127.0.0.1 master
```

## worker

### collocated

```
<MASTER_PRIVATE_IP> master
```

### non-collacated

```
<MASTER_PUBLIC_IP> master
```

# debugging envs

```bash
NCCL_DEBUG=WARN
TORCH_CPP_LOG_LEVEL=INFO
TORCH_DISTRIBUTED_DEBUG=DETAIL
```

# debugging pytorch

```python

torch.distributed.is_available() # is pytorch compiled with distributed support?

torch.distributed.is_initialized()

torch.distributed.monitored_barrier() # supported by gloo only

```
