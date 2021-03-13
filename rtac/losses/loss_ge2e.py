import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq


class SimilarityMatrix(tf.keras.layers.Layer):
  def __init__(self, n, m):
    super(SimilarityMatrix, self).__init__()
    self.n = n
    self.m = m
    self.i_eq_j_np = np.repeat(np.eye(n), m, axis=0)
    self.i_eq_j = tf.constant(self.i_eq_j_np, dtype=tf.float32)

  ''' calculates similarity matrix
  Args:
    e: Tensor [n*m, p], the e_ji from same speakers are grouped at every m vectors
      (p: projection units)
  Returns:
    s: Tensor [n*m, n]
  '''
  def call(self, e):
    n, m = self.n, self.m
    # e_i_eq_j is the colored regions in fig.1, shape [n*m, n, p]
    # centroids: - e_i_eq_j implements equation 8 and 9
    speaker_utterances = tf.reshape(e, [n, m, -1])
    i = tf.expand_dims(self.i_eq_j, -1)
    e_i_eq_j = i * tf.expand_dims(e, 1)
    centroids = tf.reduce_sum(speaker_utterances, axis=-2) - e_i_eq_j
    centroids = (i / (m - 1)) * centroids + ((1 - i) / m) * centroids

    # cosine similarity
    normalised_e_ji = tf.expand_dims(tf.nn.l2_normalize(e, -1), axis=-2)
    normalised_c_k = tf.nn.l2_normalize(centroids, -1)
    s = tf.reduce_sum(normalised_e_ji * normalised_c_k, axis=-1)

    return s


class SoftmaxLoss(tf.keras.layers.Layer):
  def __init__(self, n, m, w, b):
    super(SoftmaxLoss, self).__init__(name='softmax_loss')
    self.n = n
    self.m = m
    self.w = w
    self.b = b
    self.sm = SimilarityMatrix(n, m)
    self.i_eq_j = self.sm.i_eq_j
    self.i_eq_j_np = self.sm.i_eq_j.numpy()

  ''' calculates softmax loss on given similarity matrix
  Args:
    e: Tensor [n*m, p], the e_ji from same speakers are grouped at every m vectors
      (p: projection units)
  '''
  def call(self, e):
    s = self.sm(e)
    s = tf.abs(self.w) * s + self.b
    s_jij = tf.reduce_sum(s * self.i_eq_j, axis=-1)
    softmax = tf.math.log(tf.reduce_sum(tf.math.exp(s), axis=-1) + 1e-9)
    loss = tf.reduce_mean(-s_jij + softmax)
    return loss, s
    
  def eer(self, s):
    fpr, tpr, thresholds = roc_curve(self.i_eq_j_np.flatten(), s.flatten())           
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return np.asarray(eer).astype(np.float32)

  def accuracy(self, s):
    n, m = self.n, self.m
    pred = tf.argmax(s, axis=-1, output_type=tf.int32)
    truth = tf.constant(np.repeat(range(n), m), dtype=tf.int32)
    correct = tf.cast(tf.equal(pred, truth), tf.int32)
    accuracy = tf.reduce_sum(correct) / n / m
    return accuracy


class ContrastLoss(tf.keras.layers.Layer):
  def __init__(self, n, m):
    super(ContrastLoss, self).__init__()

  ''' calculates contrast loss on given similarity matrix
  Args:
      s: Tensor [n*m, n], similarity matrix
  '''
  def call(self, s):
    raise NotImplementedError

