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

scp -P 28495 -pr ./data/anime.tar.gz root@66.114.112.70:/root/diffusion/anime.tar.gz

scp -P 41604 -pr ./data/stanford_cars.tar.gz root@66.114.112.70:/root/diffusion/stanford_cars.tar.gz

tar -czvf ./data/stanford_cars.tar.gz ./data/stanford_cars

tar -xzvf anime.tar.gz

tar -xzvf stanford_cars.tar.gz


LOG=I:CHECKPOINT:0

---

- reduce lr
- random horizontal flip
- we shoudl not do wandb init for run.py; only do logins

---

[2024-04-14 22:44:59,020] torch.distributed.run: [WARNING] 
[2024-04-14 22:44:59,020] torch.distributed.run: [WARNING] *****************************************
[2024-04-14 22:44:59,020] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2024-04-14 22:44:59,020] torch.distributed.run: [WARNING] *****************************************

---

sudo nvidia-smi -pl 450

sudo nvidia-smi -q -d POWER

---

- attention
- group normalization
- change `IMG_SIZE, IMG_SIZE` -> `IMG_HEIGHT, IMG_WIDTH`
- do not corrupt wandb run data when doing inference
- make dataset permament on something like s3
- save intermediate sample into a folder & wandb (during training)


potential experiments
- try l2 loss
- lr decay
- attention block

---

/root/diffusion/src/plt.py:17: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig, fig_ax = plt.subplots(

---
