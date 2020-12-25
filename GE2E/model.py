import numpy as np
import tensorflow as tf


class GE2E(tf.keras.Model):
  def __init__(self, 
               num_cnn,
               cnn_filters,
               rnn_filters,
               projection_filters,
               w,
               b,
               name='ge2e'):
    super(GE2E, self).__init__(name=name)
    self.conv = []
    for i in range(num_cnn):
      filters = (i + 1) * cnn_filters
      self.conv.append(tf.keras.layers.Conv1D(
          filters=2*filters, 
          kernel_size=3, 
          strides=2,
          padding='same',
          name='cnn_%d'%i))
    self.rnn = tf.keras.layers.GRU(
        units=rnn_filters, 
        return_sequences=False, 
        recurrent_activation=tf.nn.sigmoid, 
        dropout=0.5,
        name='rnn')
    self.projection = tf.keras.layers.Dense(
        projection_filters, 
        use_bias=False, 
        name='projection')
    self.w = tf.Variable(w, name='rescale/w', dtype=tf.float32)
    self.b = tf.Variable(b, name='rescale/b', dtype=tf.float32)

  def set_shape(self, n_mels):
      self.call = tf.function(self.call, input_signature=[
          tf.TensorSpec(shape=[None, None, n_mels], dtype=tf.float32, name="ge2e_mels"), 
          tf.TensorSpec(shape=[], dtype=tf.bool, name="training"), 
      ])
      self.inference = tf.function(self.inference, input_signature=[
          tf.TensorSpec(shape=[None, None, n_mels], dtype=tf.float32, name="ge2e_mels"),
      ])
      self.n_mels = n_mels

  def call(self, x, training):
    for conv in self.conv:
      x = conv(x)
      a, b = tf.split(x, 2, axis=-1)
      x = tf.nn.tanh(a) * tf.nn.sigmoid(b)

    x = self.rnn(x, training=training)
    x = self.projection(x)

    # normalise projection as embedding vector
    x = tf.nn.l2_normalize(x, axis=-1)

    return x

  def inference(self, x):
    return tf.reduce_mean(self(x, training=False), axis=0)


