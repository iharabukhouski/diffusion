# error
# warning
# info
# debug
import os

# DEFAULT_LOGGER_LEVEL = 'DEBUG'
# LOGGER_LEVEL = os.getenv('LOG', '0') == '1':

def error(
  *args,
):

  print('[ERROR]', *args)

def info(
  *args,
):

  print('[INFO]', *args)

def debug(
  *args,
):

  if int(os.getenv('LOG', '0')) >= 1:

    print('[DEBUG]', *args)
