import torch.distributed as dist

class Distributed:

  """

  GLOO - supports only CPU
  NCCL - supports only CUDA

  """

  def __init__(
    self,
    logger,
    device,
  ):

    self.logger = logger('DISTRIBUTED')

    self.logger.debug('Init Start')

    # dist.init_process_group(
    #  backend = dist.Backend.MPI,
    #  rank = rank,
    #  world_size = world_size,
    # )

    # TODO: I need to compile torch from source on the machine that has MPI installed
    # NOTE: NCCL not supported on macos
    backend = dist.Backend.NCCL if device.is_cuda() else dist.Backend.GLOO

    dist.init_process_group(
      backend = backend
    )

    self.logger.debug('Init Finish')

  def barrier(
    self,
  ):

    dist.barrier()

  def destroy(
    self,
  ):

    self.logger.debug('Destroy Start')

    dist.destroy_process_group()

    self.logger.debug('Destroy Finish')
