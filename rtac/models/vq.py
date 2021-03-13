import tensorflow as tf


class VQ(tf.keras.layers.Layer):
  def __init__(self, 
               latent_dim=64, 
               k=512, 
               beta=0.25,
               seed=0):
    super().__init__(name='vq')
    self.k = k
    self.codebook = tf.Variable(
        name='codebook', 
        shape=[k, latent_dim], 
        initial_value=tf.keras.initializers.HeNormal(seed)(shape=[k, latent_dim]))

  ''' discretises z_e
  Args:
    z_e (Tensor): Input tensor (..., d)
  Returns:
    Tensor: Output tensor (B, T, C)
  '''
  def call(self, z_e):
    # z_e: [..., d] -> [..., 1, d]
    # codebook:             [k, d]
    expanded_ze = tf.expand_dims(z_e, -2)
    distances = tf.reduce_sum((expanded_ze - self.codebook) ** 2, axis=-1)
    # q(z|x) refers to the 2d grid in the middle in figure 1
    q_z_x = tf.argmin(distances, axis=-1, output_type=tf.int32)
    # e_k = tf.gather(params=self.codebook, indices=q_z_x)
    e_k = tf.nn.embedding_lookup(self.codebook, q_z_x)
    # passing gradient from z_q to z_e
    z_q = z_e + tf.stop_gradient(e_k - z_e)

    encodings = tf.one_hot(tf.reshape(q_z_x, [-1]), self.k)
    avg_probs = tf.reduce_mean(encodings, 0)
    perplexity = tf.math.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))

    items = {
      'z_e': z_e,
      'codebook': self.codebook,
      'distances': distances,
      'q(z|x)': q_z_x,
      'perplexity': perplexity,
      'e_k': e_k,
      'z_q': z_q,
    }
    return items

'''
from tensorflow.python.training import moving_averages
class VQ_EMA(tf.keras.layers.Layer):
  def __init__(self, 
               latent_dim=64, 
               k=512, 
               beta=0.25,
               seed=0,
               decay=0.998,
               epsilon=1e-5):
    super().__init__(name='vq')
    w = tf.keras.initializers.HeNormal(seed)(shape=[k, latent_dim])
    self._w = tf.Variable(
        name='codebook', 
        shape=[k, latent_dim], 
        initial_value=w)
    self._ema_cluster_size = tf.Variable(
        name='ema_cluster_size', 
        shape=[k], 
        initial_value=tf.keras.initializers.Zeros()(shape=[k, latent_dim]))
    self._ema_w = tf.Variable(
        name='ema_w', 
        shape=[k, latent_dim], 
        initial_value=w)

    self._embedding_dim = latent_dim
    self._num_embeddings = k
    self._decay = decay
    self._commitment_cost = beta
    self._epsilon = epsilon
    self.trainable = trainable

    self.codebook = self._w
      
  def __call__(self, inputs):
    # _, t, f = inputs.get_shape().as_list()
    with tf.control_dependencies([inputs]):
      w = self._w.read_value()
    input_shape = tf.shape(inputs)

    with tf.control_dependencies([
        tf.Assert(tf.equal(input_shape[-1], self._embedding_dim),
                  [input_shape])]):
      flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])

    distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True)
                 - 2 * tf.matmul(flat_inputs, w)
                 + tf.reduce_sum(w ** 2, 0, keepdims=True))

    encoding_indices = tf.argmax(- distances, 1)
    encodings = tf.one_hot(encoding_indices, self._num_embeddings)
    encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
    quantized = self.quantize(encoding_indices)
    e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)

    if self.trainable:
      updated_ema_cluster_size = moving_averages.assign_moving_average(
          self._ema_cluster_size, tf.reduce_sum(encodings, 0), self._decay)
      dw = tf.matmul(flat_inputs, encodings, transpose_a=True)
      updated_ema_w = moving_averages.assign_moving_average(self._ema_w, dw,
                                                            self._decay)
      n = tf.reduce_sum(updated_ema_cluster_size)
      updated_ema_cluster_size = (
          (updated_ema_cluster_size + self._epsilon)
          / (n + self._num_embeddings * self._epsilon) * n)

      normalised_updated_ema_w = (
          updated_ema_w / tf.reshape(updated_ema_cluster_size, [1, -1]))
      with tf.control_dependencies([e_latent_loss]):
        update_w = tf.assign(self._w, normalised_updated_ema_w)
        with tf.control_dependencies([update_w]):
          loss = self._commitment_cost * e_latent_loss

    else:
      loss = self._commitment_cost * e_latent_loss

    quantized = inputs + tf.stop_gradient(quantized - inputs)
    avg_probs = tf.reduce_mean(encodings, 0)
    perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))

    tf.summary.scalar('e_latent_loss', e_latent_loss)
    tf.summary.histogram('inputs', inputs)
    tf.summary.histogram('codebook', self._w)
    tf.summary.histogram('ema_cluster_size', self._ema_cluster_size)
    tf.summary.histogram('ema_w', self._ema_w)
    tf.summary.histogram('distances', distances)
    tf.summary.histogram('q(z|x)', encoding_indices)
    tf.summary.histogram('e_k', quantized)
    self.loss = loss
    return quantized

  @property
  def embeddings(self):
    return self._w

  def compute_loss(self):
    return self.loss

  def quantize(self, encoding_indices):
    with tf.control_dependencies([encoding_indices]):
      w = tf.transpose(self.embeddings.read_value(), [1, 0])
    return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)
'''

class Jitter(tf.keras.layers.Layer):
  def __init__(self, move_prob=0.12):
    super().__init__(name='jitter')
    self.prob = [[move_prob / 2, 1 - move_prob, move_prob / 2]]

  '''at time t, either use sample at t-1, t, or t+1, using gather
  Args:
    x (Tensor): Input tensor (B, T, C)
  Returns:
    Tensor: Output tensor (B, T, C)
  '''
  def call(self, x, training):
    if not training or self.prob[0][0] == 0:
      return x

    B_T_C = tf.shape(x)
    x = tf.reshape(x, [-1, B_T_C[-1]])
    BT = tf.shape(x)[0]

    # {0, 1, 2} -> {-1, 0, 1}
    # categories = tf.distributions.Categorical(probs=self.prob)
    # move_step = categories.sample(BT) - 1
    move_step = tf.random.categorical(self.prob, BT, dtype=tf.int32)
    move_step = tf.squeeze(move_step, 0) - 1

    indices = tf.range(BT, dtype=tf.int32)
    indices += move_step
    indices += 2 * tf.cast(indices < 0, tf.int32)
    indices -= 2 * tf.cast(indices >= BT, tf.int32)

    x = tf.gather(x, indices)
    x = tf.reshape(x, B_T_C)
    return x


class ResBlock(tf.keras.layers.Layer):
  def __init__(self, 
               filters, 
               kernel_size, 
               strides,
               **kwargs):
    super().__init__(**kwargs)
    self.start = tf.keras.layers.Conv1D(
        filters, 
        kernel_size=kernel_size, 
        strides=strides,
        padding='same',
        name='start')
    self.res = tf.keras.layers.Conv1D(
        filters, 
        kernel_size=kernel_size, 
        padding='same',
        name='res')
    self.conv = tf.keras.layers.Conv1D(
        filters*2, 
        kernel_size=kernel_size, 
        padding='same',
        name='conv')

  def call(self, x):
    x = self.start(x)
    res = self.res(x)
    x = self.conv(x)
    a, b = tf.split(x, 2, axis=-1)
    x = tf.nn.sigmoid(a) * tf.nn.tanh(b)
    x = (x + res) * (0.5 ** 0.5)
    return x


class Encoder(tf.keras.Model):
  '''encode + vq + jitter'''
  def __init__(self, 
               n_blocks=4,
               filters=256, 
               kernel_size=3,
               name='encoder',
               vq_args={'latent_dim': 64, 'k': 512, 'beta': 0.25, 'seed': 0},
               jitter_args={'move_prob': 0.12}):
    super().__init__(name=name)
    layers = []
    for i in range(n_blocks):
      layers += [
        ResBlock(filters=filters, 
                 kernel_size=kernel_size, 
                 strides=1,
                 name='resblock_%d'%i),
        tf.keras.layers.ReLU()
      ]
    layers += [
      tf.keras.layers.Conv1D(vq_args.get('latent_dim'), 
                             kernel_size=1, 
                             padding='same')
    ]
    self.encoder = tf.keras.Sequential(layers)
    self.vq = VQ(**vq_args)
    self.jitter = Jitter(**jitter_args)

  def call(self, x, training=True):
    z_e = self.encoder(x, training=training)
    items = self.vq(z_e)
    items.update({'z_q': self.jitter(items['z_q'], training=training)})
    return items

