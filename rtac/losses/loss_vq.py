
import tensorflow as tf


class VQLoss(tf.keras.layers.Layer):
  def __init__(self, beta=0.25, name='vq_loss'):
    super().__init__(name=name)
    self.beta = beta
    
  def call(self, z_e, e_k):
    axis = list(range(1, len(z_e.shape)))
    vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - e_k) ** 2, axis=axis)
    commitment_loss = tf.reduce_mean((z_e - tf.stop_gradient(e_k)) ** 2, axis=axis)
    return vq_loss + self.beta * commitment_loss

