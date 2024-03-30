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

    print(f'[E] [{self.namespace}:{self.rank}]', *args)

  def info(
    self,
    *args,
  ):

    # TODO: only if log level is 1; default log level should be 1

    print(f'[I] [{self.namespace}:{self.rank}]', *args)

  def debug(
    self,
    *args,
  ):

    # TODO: only if log level is 2

    if int(os.getenv('LOG', '0')) >= 1:

      print(f'[D] [{self.namespace}:{self.rank}]', *args)
