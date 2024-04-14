# error
# warning
# info
# debug
import os

# DEFAULT_LOGGER_LEVEL = 'DEBUG'
# LOGGER_LEVEL = os.getenv('LOG', '0') == '1':


def info(
  *args,
):

  print('[INFO]', *args)

def debug(
  *args,
):

  if int(os.getenv('LOG', '0')) >= 1:

    print('[DEBUG]', *args)

class Logger:

  def __init__(
    self,
    rank,
    namespace,
  ):

    self.rank = rank
    self.namespace = namespace

  def error(
    self,
    *args,
  ):

    # TODO: only if log level is 0

    print(f'[E] [{self.namespace}{self.repr_rank()}]', *args)

  def info(
    self,
    *args,
  ):

    # TODO: only if log level is 1; default log level should be 1

    if self.rank == 0:

      print(f'[I] [{self.namespace}{self.repr_rank()}]', *args)

  def debug(
    self,
    *args,
  ):

    # TODO: only if log level is 2

    if int(os.getenv('LOG', '0')) >= 1:

      print(f'[D] [{self.namespace}{self.repr_rank()}]', *args)

  def repr_rank(
    self,
  ):

    return f':{self.rank}' if self.rank is not None else ''
