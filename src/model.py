import math
import torch
from torch import nn
import config
import logger

class Block(nn.Module):

  def __init__(
      self,
      number_of_input_channels,
      number_of_output_channels,
      timestep_embedding_dimentionality, # positional encoding
      down = False,
  ):

    super().__init__()

    self.name = f'{number_of_input_channels} -> {number_of_output_channels}'

    # TODO: I still do not understand why we embed timestep the second time here; I feel like we might not need it
    # Positional encoding
    self.timesteps_embedding = nn.Linear(
      in_features = timestep_embedding_dimentionality,
      out_features = number_of_output_channels,
    )

    # Down Sampling
    if down:

      self.conv1 = nn.Conv2d(
        in_channels = number_of_input_channels,
        out_channels = number_of_output_channels,
        kernel_size = 3,
        padding = 1,
      )

      # Down Sampling
      self.transform = nn.Conv2d(
        in_channels = number_of_output_channels,
        out_channels = number_of_output_channels,
        kernel_size = 4,
        stride = 2,
        padding = 1,
      )

    # Up Sampling
    else:

      self.conv1 = nn.Conv2d(
        in_channels = 2 * number_of_input_channels,
        out_channels = number_of_output_channels,
        kernel_size = 3,
        padding = 1,
      )

      # Up Sampling
      self.transform = nn.ConvTranspose2d(
        in_channels = number_of_output_channels,
        out_channels = number_of_output_channels,
        kernel_size = 4,
        stride = 2,
        padding = 1,
      )

    self.conv2 = nn.Conv2d(
      in_channels = number_of_output_channels,
      out_channels = number_of_output_channels,
      kernel_size = 3,
      padding = 1,
    )

    # Batch Normalization Layer
    self.bnorm1 = nn.BatchNorm2d(
      number_of_output_channels,
    )

    # Batch Normalization Layer
    self.bnorm2 = nn.BatchNorm2d(
      number_of_output_channels,
    )

    # Non-linearity Layer
    self.relu = nn.ReLU()

  def forward(
    self,
    x,
    timesteps, # t of (BATCH_SIZE, TIMESTEP_EMBEDDING_DIMENTIONALITY)
  ):

    h = self.bnorm1(self.relu(self.conv1(x)))

    timesteps_embeddings = self.relu(self.timesteps_embedding(timesteps))

    timesteps_embeddings = timesteps_embeddings[(..., None, None)]

    h = h + timesteps_embeddings

    h = self.bnorm2(self.relu(self.conv2(h)))

    # Down / Up Sampling
    return self.transform(h)

class SinusoidalPositionalEmbeddings(nn.Module):

  def __init__(
      self,
      dimentionality,
  ):

    super().__init__()

    self.dimentionality = dimentionality

  def forward(
    self,
    timesteps, # t of (BATCH_SIZE)
  ):

    half_dimentionality = self.dimentionality // 2
    embeddings = math.log(10000) / (half_dimentionality - 1)
    embeddings = torch.exp(torch.arange(half_dimentionality) * -embeddings)
    embeddings = timesteps[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim = -1)

    return embeddings # of (BATCH_SIZE, TIMESTEP_EMBEDDING_DIMENTIONALITY)

class UNet(nn.Module):

  def __init__(
    self,
  ):

    super().__init__()

    # Time embeddings
    # SEE: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1
    self.timesteps_embedding = nn.Sequential(
      SinusoidalPositionalEmbeddings(
        config.TIMESTEP_EMBEDDING_DIMENTIONALITY,
      ),
      nn.Linear(
        config.TIMESTEP_EMBEDDING_DIMENTIONALITY,
        config.TIMESTEP_EMBEDDING_DIMENTIONALITY,
      ),
      nn.ReLU(),
    )

    # Initial Projection
    self.initial_projection = nn.Conv2d(
      in_channels = config.CHANNELS[0],
      out_channels = config.CHANNELS[1],
      kernel_size = 3,
      padding = 1,
    )

    # Down Sample
    self.downs = nn.ModuleList(
      [
        Block(
          config.CHANNELS[i],
          config.CHANNELS[i + 1],
          config.TIMESTEP_EMBEDDING_DIMENTIONALITY,
          down = True,
        )
        for i in range(1, len(config.CHANNELS) - 1)
      ] 
    )

    # Up Sample
    self.ups = nn.ModuleList(
      [
        Block(
          config.CHANNELS[i * -1],
          config.CHANNELS[i * -1 - 1],
          config.TIMESTEP_EMBEDDING_DIMENTIONALITY,
        )
        for i in range(1, len(config.CHANNELS) - 1)
      ]
    )

    # Final Projection
    self.final_projection = nn.Conv2d(
      in_channels = config.CHANNELS[1],
      out_channels = config.CHANNELS[0],
      kernel_size = 1, # TODO: ???
    )

  def forward(
    self,
    images_at_timesteps, # x_t of (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    timesteps, # t of (BATCH_SIZE / ???)
  ):

    """
    Returns predicted noise of shape (IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    """

    # Create timesteps embeddings
    embedded_timesteps = self.timesteps_embedding(timesteps) # (BATCH_SIZE, TIMESTEP_EMBEDDING_DIMENTIONALITY)

    # Initial convolution
    x = self.initial_projection(images_at_timesteps)

    residuals = []

    for down in self.downs:

      x = down(
        x,
        embedded_timesteps,
      )

      residuals.append(x)

    for up in self.ups:

      residual = residuals.pop()

      x = torch.cat(
        (
          x,
          residual,
        ),
        dim = 1,
      )

      x = up(
        x,
        embedded_timesteps,
      )

    noise_at_timestep = self.final_projection(x) # epsilon_t

    return noise_at_timestep

def print_model(
  model,
):

  parameters = sum(p.numel() for p in model.parameters())
  logger.info(f'[MODEL] Parameters: {parameters:,}')
  logger.info(f'[MODEL] Parameters (MB): {(parameters * 4 / 1024 ** 2):.6}')
  logger.info('[MODEL] Device:', next(model.parameters()).device)
  logger.debug('[MODEL] Architecture:', model)

class UNet2(nn.Module):

  def __init__(
    self,
  ):

    super().__init__()

    self.layer = nn.Linear(config.IMG_SIZE, config.IMG_SIZE, bias = False)

  def forward(
    self,
    x,
    y,
  ):

    return self.layer(x)
