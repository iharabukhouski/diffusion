import torch.distributed as dist

# SEE: https://pytorch.org/docs/stable/distributed.html
class Distributed:

  """

  GLOO - supports only CPU
  NCCL - supports only CUDA
  MPI - only supported if pytorch was compiled on a machine with mpi installed

  """

  def __init__(
    self,
    logger,
    device,
  ):

    self.logger = logger('DISTRIBUTED')

    self.logger.debug('Init Start')

    if dist.is_gloo_available():

      self.logger.debug('GLOO available')

    if dist.is_nccl_available():

      self.logger.debug('NCCL available')

    if dist.is_mpi_available():

      self.logger.debug('MPI available')

    if dist.is_torchelastic_launched():

      self.logger.debug('Torch Elastic')

    # dist.init_process_group(
    #  backend = dist.Backend.MPI,
    #  rank = rank,
    #  world_size = world_size,
    # )

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
