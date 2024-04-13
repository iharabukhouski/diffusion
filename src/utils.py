from time import perf_counter
from contextlib import ContextDecorator

class perf(ContextDecorator):

  def __init__(
    self,
    logger,
    message,
  ):

    self.logger = logger
    self.message = message

  def __enter__(
    self,
  ):

      self.start = perf_counter()

      self.logger.debug(f'{self.message} Start')

      return self

  def __exit__(
    self,
    type,
    value,
    traceback,
  ):

      self.end = perf_counter()

      self.logger.debug(f'{self.message} Finish: {self.end - self.start:.4f}s')
