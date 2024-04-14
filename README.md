- [Dataset](#dataset)
  - [Clone](#clone)
- [Model](#model)
- [Config](#config)
  - [Dependencies](#dependencies)
  - [.env Files](#env-files)
  - [Envs](#envs)
- [Training](#training)
  - [Local](#local)
  - [Multi-GPU](#multi-gpu)
  - [Multi-Node Multi-GPU](#multi-node-multi-gpu)
    - [Node 0 (Master)](#node-0-master)
    - [Node 1 (Worker)](#node-1-worker)
- [Inference](#inference)
- [Profiling](#profiling)

# Dataset

[Dataset](./docs/dataset.md)

## Clone

```bash
touch ~/.no_auto_tmux
apt-get install git-lfs
git lfs install
mkdir data
cd data
git clone https://huggingface.co/datasets/iharabukhouski/stanford_cars
```

```bash
scp -P 45192 -pr ./data/anime.tar.gz root@45.23.135.240:/root/diffusion/data/anime.tar.gz
```

# Model

[Model](./docs/model.md)

# Config

## Dependencies

```bash
pip3 install .
```

## .env Files

```
WANDB_API_KEY = <W&B API KEY>
```

## Envs
- `WANDB` - disable / enable WANDB (default is 1)
- `MPS` - use mps device
- `CUDA` - use cuda device
- `CPU` - user cpu device
- `RUN` - wandb run_id
- `LOG` - 1 for debugging
- `PERF` - 1 for performance
- `BS` - batch size
- `DS` - dataset size
- `EPOCHS` - number of epochs
- `GPUS` - number of gpus
- `CPUS` - number of cpus

# Training

## Local

```bash
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--node_rank=0 \
--master_addr=localhost \
--master_port=40723 \
./train.py
```

## Multi-GPU

```bash
torchrun \
--nnodes=1 \
--nproc_per_node=2 \
--node_rank=0 \
--master_addr=localhost \
--master_port=40723 \
./train.py
```

## Multi-Node Multi-GPU


### Node 0 (Master)

```bash
torchrun \
--nnodes=2 \
--nproc_per_node=2 \
--node_rank=0 \
--master_addr=master \
--master_port=40723 \
./train.py
```

### Node 1 (Worker)

```bash
torchrun \
--nnodes=2 \
--nproc_per_node=2 \
--node_rank=1 \
--master_addr=master \
--master_port=40723 \
./train.py
```

# Inference

```bash
MPS=1 RUN=<WANDB_RUN_ID> ./run.py
```

# Profiling

[Profiling](./docs/profiling.md)

---

scp -P 41542 -pr ./data/anime.tar.gz root@174.31.93.199:/root/diffusion/anime.tar.gz

tar -czvf anime.tar.gz directory

tar -xzvf anime.tar.gz


LOG=I:CHECKPOINT:0

---

- reduce lr
- random horizontal flip
