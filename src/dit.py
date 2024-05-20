# [DiT](https://github.com/facebookresearch/DiT/blob/main/models.py)
# [VAE](https://github.com/CompVis/latent-diffusion#pretrained-autoencoding-models)

import torch
import torch.nn as nn
import math
import numpy as np

def modulate(
  x,
  shift, # beta
  scale, # gamma
):

  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):

  def __init__(
    self,
    hidden_size,
    frequency_embedding_size = 256,
  ):

    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(
        frequency_embedding_size,
        hidden_size,
        bias = True,
      ),
      nn.SiLU(),
      nn.Linear(
        hidden_size,
        hidden_size,
        bias = True,
      ),
    )
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(
    t,
    dim,
    max_period = 10000,
  ):

    half = dim // 2
    freqs = torch.exp(
      -math.log(max_period) * torch.arange(start = 0, end = half, dtype = torch.float32) / half
    ).to(device = t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim = -1)
    if dim % 2:
      embedding = torch.cat([embedding, torch.zeros_like[embedding[:, :1]]], dim = -1)
    return embedding

  def forward(
    self,
    t,
  ):

    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)

    return t_emb

class LabelEmbedder(nn.Module):

  def __init__(
    self,
    num_classes,
    hidden_size,
    dropout_prob,
  ):

    super().__init__()

    use_cfg_embedding = dropout_prob > 0
    self.embedding_table = nn.Embedding(
      num_classes + use_cfg_embedding,
      hidden_size,
    )
    self.num_classes = num_classes
    self.dropout_prob = dropout_prob

  def token_drop(
    self,
    labels,
    force_drop_ids = None,
  ):

    if force_drop_ids is None:

      drop_ids = torch.rand(
        labels.shape[0],
        device = labels.device
      ) < self.dropout_prob

    else:

      drop_ids = force_drop_ids == 1

    labels = torch.where(
      drop_ids,
      self.num_classes,
      labels,
    )

    return labels
  
  def forward(
    self,
    labels,
    train,
    force_drop_ids = None,
  ):

    use_dropout = self.dropout_prob > 0

    if (train and use_dropout) or (force_drop_ids is not None):

      labels = self.token_drop(labels, force_drop_ids)

    embeddings = self.embedding_table(labels)

    return embeddings

# class Attention(nn.Module):
#     fused_attn: Final[bool]

#     def __init__(
#             self,
#             dim: int,
#             num_heads: int = 8,
#             qkv_bias: bool = False,
#             qk_norm: bool = False,
#             attn_drop: float = 0.,
#             proj_drop: float = 0.,
#             norm_layer: nn.Module = nn.LayerNorm,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.fused_attn = use_fused_attn()

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
#         q, k = self.q_norm(q), self.k_norm(k)

#         if self.fused_attn:
#             x = F.scaled_dot_product_attention(
#                 q, k, v,
#                 dropout_p=self.attn_drop.p if self.training else 0.,
#             )
#         else:
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = attn @ v

#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

class Attention(nn.Module):

  def __init__(
    self,
    dim, # dimentionality of x
    num_heads = 8,
    qkv_bias = False,
  ):

    super().__init__()

    assert dim % num_heads == 0, '"dim" must be devisible by "num_heads"'

    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    self.qkv = nn.Linear(
      dim,
      dim * 3,
      bias = qkv_bias,
    )
    self.proj = nn.Linear(
      dim,
      dim,
    )

  def forward(
    self,
    x,
  ):

    # B - batch size
    # N - sequence length???
    # C - channels??? / d_embed / dimentionality of x
    B, N, C = x.shape

    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

    # fused version F.scaled_dot_product_attention
    attn = q @ k
    attn = attn.transpose(-2, -1)
    attn = attn.softmax(dim = -1)

    x = attn @ v
    x = x.transpose(1, 2)
    x = x.reshape(B, N, C)

    x = self.proj(x)

    return x

# class Mlp(nn.Module):
#     """ MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#     def __init__(
#             self,
#             in_features,
#             hidden_features=None,
#             out_features=None,
#             act_layer=nn.GELU,
#             norm_layer=None,
#             bias=True,
#             drop=0.,
#             use_conv=False,
#     ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         bias = to_2tuple(bias)
#         drop_probs = to_2tuple(drop)
#         linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

#         self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop_probs[0])
#         self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
#         self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
#         self.drop2 = nn.Dropout(drop_probs[1])

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.norm(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x

class MLP(nn.Module):

  def __init__(
    self,
    in_features,
    hidden_features,
    out_features,
    activation,
  ):

    super().__init__()

    self.fc1 = nn.Linear(
      in_features,
      hidden_features,
      bias = True,
    )
    self.act = activation()
    self.fc2 = nn.Linear(
      hidden_features,
      out_features,
    )

  def forward(
    self,
    x,
  ):

    x = self.fc1(x)
    x = self.act(x)
    x = self.fc2(x)

    return x


class DiTBlock(nn.Module):

  def __init__(
    self,
    dim,
    num_heads,
    mlp_ratio,
  ):

    super().__init__()

    self.norm1 = nn.LayerNorm(
      dim,
      elementwise_affine = False, # turns off beta & gamma
      eps = 1e-6
    )

    # aka Multi-Head Self-Attention
    self.attn = Attention(
      dim = dim,
      num_heads = num_heads,
      qkv_bias = True,
    )

    self.norm2 = nn.LayerNorm(
      dim,
      elementwise_affine = False, # turns off beta & gamma
      eps = 1e-6,
    )

    # aka Pointwise Feed Forward in DiT diagram
    self.mlp = MLP(
      in_features = dim,
      hidden_features = int(dim * mlp_ratio),
      out_features = dim,
      activation = lambda: nn.GELU(approximate = 'tanh'),
    )

    # aka MLP in DiT diagram
    self.adaLN_modulation = nn.Sequential(
      nn.SiLU(),
      nn.Linear(
        dim,
        6 * dim,
        bias = True,
      ),
    )

  def forward(
    self,
    x,
    c,
  ):

    # shift - beta
    # scale - gamma
    # gate - alpha
    # msa - multi-head self attention
    # mlp - multi-layer perceptron
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim = 1)
    x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
    x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
    return x

class FinalLayer(nn.Module):

  def __init__(
    self,
    hidden_size,
    patch_size,
    out_channels,
  ):

    super().__init__()

    self.norm_final = nn.LayerNorm(
      hidden_size,
      elemetwise_affine = False,
      eps = 1e-6,
    )
    self.linear = nn.Linear(
      hidden_size,
      patch_size * patch_size * out_channels,
      bias = True,
    )
    self.adaLN_modulation = nn.Sequential(
      nn.SiLU(),
      nn.Linear(
        hidden_size,
        2 * hidden_size,
        bias = True,
      )
    )

  def forward(
    self,
    x,
    c,
  ):
    
    shift, scale = self.adaLN_modulation(c).chunk(2, dim = 1)
    x = modulate(self.norm_final(x), shift, scale)
    x = self.linear(x)

    return x

class DiT(nn.Module):

  def __init__(
    self,
    input_size = 32, # ???
    patch_size = 2,
    in_channels = 4, # ???
    hidden_size = 1152, # ???
    depth = 28, # ???
    num_heads = 16,
    mlp_ratio = 4,
    class_dropout_prob = 0.1, # 10%
    num_classes = 1000,
    learn_sigma = True,
  ):

    super().__init__()

    self.learn_sigma = learn_sigma
    self.in_channels = in_channels
    self.out_channels = in_channels * 2 if learn_sigma else in_channels
    self.patch_size = patch_size
    self.num_heads = num_heads

    self.x_embedder = PatchEmbedder(
      input_size,
      patch_size,
      in_channels,
      hidden_size,
      bias = True,
    )
    self.t_embedder = TimestepEmbedder(
      hidden_size
    )
    self.y_embedder = LabelEmbedder(
      num_classes,
      hidden_size,
      class_dropout_prob,
    )

    num_patches = self.x_embedder.num_patches

    self.pos_embed = nn.Parameter(
      torch.zeros(
        1,
        num_patches,
        hidden_size,
      ),
      requires_grad = False,
    )

    self.blocks = nn.ModuleList(
      [
        DiTBlock(
          hidden_size,
          num_heads,
          mlp_ratio = mlp_ratio,
        ) for _ in range(depth)
      ]
    )
    self.final_layer = FinalLayer(
      hidden_size,
      patch_size,
      self.out_channels,
    )
    self.initialize_weights()

  def initialize_weights(
    self,
  ):

    def _basic_init(module):

      if isinstance(module, nn.Linear):

        torch.nn.init.xavier_uniform_(module.weight)

        if module.bias is not None:

          nn.init.constant_(module.bias, 0)

    self.apply(_basic_init)

    # Initialize (and freeze) pos_embed by sin-cos embedding
    pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
    self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    w = self.x_embedder.proj.weight.data
    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    nn.init.constant_(self.x_embedder.proj.bias, 0)

    # Initialize label embedding table
    nn.init.normal_(self.y_embedder.embedding_table.weight, std = 0.02)

    # Initialize timestep embedding MLP
    nn.init.normal_(self.t_embedder.mlp[0].weight, std = 0.02)
    nn.init.normal_(self.t_embedder.mlp[2].weight, std = 0.02)

    # Zero-out adaLN modulation layers in DiT blocks
    for block in self.blocks:

      nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
      nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    # Zero-out output layers
    nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
    nn.init.constant_(self.final_layer.linear.weight, 0)
    nn.init.constant_(self.final_layer.linear.bias, 0)

  def unpatchify(
    self,
    x,
  ):

    c = self.out_channels
    p = self.x_embedder.patch_size[0]
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape = (x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc-> nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs

  #
  def forward(
    self,
    x, # (N, C, H, W) latent representation
    t, # (N, )
    y, # (N, )
  ):

    x = self.x_embedder(x) + self.pos_embed
    t = self.t_embedder(t)
    y = self.y_embedder(y, self.training)
    c = t + y # conditioning

    for block in self.blocks:

      x = block(x, c)

    x = self.final_layer(x, c)

    x = self.unpatchify(x)

def get_2d_sincos_pos_embed(
  embed_dim,
  grid_size,
  cls_token = False,
  extra_tokens = 0,
):

  grid_h = np.arange(
    grid_size,
    dtype = np.float32,
  )
  grid_w = np.arange(
    grid_size,
    dtype = np.float32,
  )
  grid = np.meshgrid(
    grid_w,
    grid_h,
  )
  grid = np.stack(
    grid,
    axis = 0,
  )

  grid = grid.reshape(
    [
      2,
      1,
      grid_size,
      grid_size,
    ]
  )
  pos_embed = get_2d_sincons_pos_embed_from_grid(
    embed_dim,
    grid,
  )

  if cls_token and extra_tokens > 0:

    pos_embed = np.concatenate(
      [
        np.zeros(
          [
            extra_tokens,
            embed_dim,
          ]
        ),
        pos_embed,
      ],
      axis = 0,
    )

  return pos_embed

def get_2d_sincos_pos_embed_from_grid(
  embed_dim,
  grid,
):
  
  assert embed_dim % 2 == 0

  emb_h = get_1d_sincos_pos_embed_from_grid(
    embed_dim // 2,
    grid[0],
  )
  emb_w = get_1d_sincos_pos_embed_from_grid(
    embed_dim // 2,
    grid[1],
  )

  emb = np.concatenate(
    [
      emb_h,
      emb_w,
    ],
    axis = 1,
  )

  return emb

def get_1d_sincos_pos_embed_from_grid(
  embed_dim,
  pos,
):

  assert embed_dim % 2 == 0

  omega = np.arange(embed_dim // 2, dtype = np.float64)
  omega /= embed_dim / 2.
  omega = 1. / 10000 ** omega

  pos = pos.reshape(-1)
  out = np.einsum('m,d->md', pos, omega)

  emb_sin = np.sin(out)
  emb_cos = np.cos(out)

  emb = np.concatenate(
    [
      emb_sin,
      emb_cos,
    ],
    axis = 1
  )

  return emb

# TODO:
# - Layer Norm
# - Adaptive Layer Norm
# - Implement Attention Mechanism
# - Implement MLP
