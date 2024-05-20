#! /usr/bin/env python3

# [clip by openai](https://github.com/openai/CLIP?tab=readme-ov-file)
# [autoencoder paper](https://arxiv.org/pdf/2112.10752)
# [DiT pytorch implementation](https://github.com/facebookresearch/DiT/blob/main/models.py)
# [anime dataset](https://gwern.net/danbooru2021#danbooru2018)
# [Diffusion Model / LDM](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
# [VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)

# - [clip-g/14](https://github.com/mlfoundations/open_clip?tab=readme-ov-file#openclip)
# [CLIP ViT-bigG/14](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
# - [clip-l/14](https://github.com/mlfoundations/open_clip?tab=readme-ov-file#openclip)
# [CLIP ViT-L/14](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L38C18-L38C141)
# - [t5 1.1 xxl](https://github.com/google-research/t5x/blob/main/docs/models.md#t5-11-checkpoints)
# - caption
# - timestep
# - noised latent
# - sinusoidal encoding
# - patching
# - unpatching
# - mlp
# - linear
# - positional embedding
# - modulation
# - mm-dit-block
# - silu
# - layernorm
# - attention

# c - is encoded text conditioning

import torch
from torch import nn

torch.manual_seed(0)

LEARNING_RATE = 10 ** -4 # 0.0001
GLOBAL_BATCH_SIZE = 1024
CHECKPOINT_FREQUENCY = 100
NUMBER_OF_ATTENTION_BLOCKS = 15
LATENT_DIMENTIONALITY = 18
STEPS = 400_000
IMG_HEIGHT = 256
IMG_WIDTH = 256


caption = 'cat on a table'

c_clip_g_14_ctxt = torch.randn(1280, 77)
c_clip_l_14_ctxt = torch.randn(768, 77)
c_zeros_ctx = torch.zeros(2048, 77)

c_clip_ctxt = torch.cat(
  (
    c_clip_g_14_ctxt,
    c_clip_l_14_ctxt,
    c_zeros_ctx,
  ),
  dim = 0,
)

c_t5_ctxt = torch.rand(4096, 77)

c_ctxt = torch.cat(
  (
    c_clip_ctxt,
    c_t5_ctxt,
  ),
  dim = 1,
)

c_vec = 0 # pooled

a = torch.randn((3, 5))

print(c_ctxt.shape)

# linear
# mlp

def patching():

  pass

def unpatching():

  pass

class MMDiTBlock(nn.Module):

  def __init__(
    self,
  ):

    self.silu = nn.SiLU()
    self.linear1_y = nn.Linear()
    self.linear2_y = nn.Linear()

    self.layernorm1_text = nn.LayerNorm()
    self.linear1_text = nn.Linear()
    self.linear2_text = nn.Linear()
    self.layernorm2_text = nn.LayerNorm()
    self.mlp_text = 

    self.layernorm1_image = nn.LayerNorm()
    self.linear1_image = nn.Linear()
    self.linear2_image = nn.Linear()
    self.layernorm2_image = nn.LayerNorm()
    self.mlp_image = 

    self.attention = nn.MultiheadAttention()

  def forward(
    self,
    c,
    x,
    y,
  ):

    pass

class SD3(nn.Module):

  def __init__(
    self,
  ):

    # c
    self.linear1 = nn.Linear()

    # c_vec
    self.mlp1 = 0

    # t
    self.mlp2 = 0
    self.sinusoidal_encoding = 0

    # x
    self.linear2 = nn.Linear()
    self.positional_embedding = 0

    self.mmditblock = MMDiTBlock()

    self.modulation = 0
    self.linear = nn.Linear()

  def forward(
    self,
    noised_latent_patched,
    c_ctxt,
    c_vec,
    t,
  ):

    x = self.positional_embedding(noised_latent_patched) + self.linear2(noised_latent_patched)

    c = self.linear(c_ctxt)
    y = self.mlp1(c_vec) + self.mlp2(self.sinusoidal_encoding(t))

    z = self.mmditblock(
      c,
      x,
      y,
    )

    return unpatching(self.modulation(z))


# TODO
# - read about MLP
# - read about positional embedding
# - run t5
# - run open clip
# - read about modulatoin
# - read vae paper
