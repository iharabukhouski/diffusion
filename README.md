- [Dataset](#dataset)
  - [Clone](#clone)
- [Model](#model)
- [Config](#config)
  - [Dependencies](#dependencies)
  - [.env Files](#env-files)
  - [Envs](#envs)
- [Training](#training)
- [Inference](#inference)
- [Profiling](#profiling)

# Dataset

[Dataset](./docs/dataset.md)

## Clone

```bash
apt-get install git-lfs
git lfs install
mkdir data
cd data
git clone https://huggingface.co/datasets/iharabukhouski/stanford_cars
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
- `RUN` - wandb run_id
- `LOG` - 1 for debugging
- `PERF` - 1 for performance
- `BS` - batch size
- `DS` - dataset size
- `EPOCHS` - number of epochs

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
