- [Dataset](#dataset)
  - [Clone](#clone)
- [NVIDIA](#nvidia)
- [Model](#model)
- [Config](#config)
  - [Dependencies](#dependencies)
  - [.env Files](#env-files)
  - [Envs](#envs)
- [Training](#training)
- [Inference](#inference)
- [Profiling](#profiling)
- [FLOPS Counter](#flops-counter)
- [Lambda Labs](#lambda-labs)
  - [Bandwidth](#bandwidth)

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

# NVIDIA

```bash
watch -n0.25 nvidia-smi
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

```bash
MPS=1 RUN=<WANDB_RUN_ID> ./train.py
```

# Inference

```bash
MPS=1 RUN=<WANDB_RUN_ID> ./run.py
```

# Profiling

[Profiling](./docs/profiling.md)

# FLOPS Counter
https://pytorch.org/tnt/stable/utils/generated/torchtnt.utils.flops.FlopTensorDispatchMode.html
https://pastebin.com/AkvAyJBw
https://gist.github.com/soumith/5f81c3d40d41bb9d08041431c656b233

# Lambda Labs

## Bandwidth

https://beta-docs.lambdalabs.com/cloud/network-bandwidth
