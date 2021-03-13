import numpy as np
import tensorflow as tf
import os
from pathlib import Path

def find_files(root, suffix='.wav', depth=None):
  root = Path(root)
  files = []
  if depth is None:
    depth = [1, 2, 3]
  elif type(depth) == int:
    depth = [depth]
  for d in depth:
    pattern = '/'.join('*' * d) + suffix
    files += map(str, root.glob(pattern))
  return files
  
def suppress_tf_warning():
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def print_checkpoint(path):
  from tensorflow.contrib.framework.python.framework import checkpoint_utils
  var_list = checkpoint_utils.list_variables(path)
  for v in var_list:
      print(v)

