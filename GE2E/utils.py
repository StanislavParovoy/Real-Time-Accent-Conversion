import numpy as np
import tensorflow as tf
import os

def suppress_tf_warning():
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def print_checkpoint(path):
  from tensorflow.contrib.framework.python.framework import checkpoint_utils
  var_list = checkpoint_utils.list_variables(path)
  for v in var_list:
      print(v)

