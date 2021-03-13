
import numpy as np
import tensorflow as tf

from tensorflow_tts.utils import WeightNormalization
from tensorflow_tts.models.melgan import (get_initializer,
  TFReflectionPad1d, TFConvTranspose1d, TFResidualStack, TFMelGANGenerator)


def if_add_gc(layer):
  return isinstance(layer, tf.keras.layers.Conv1D) or \
         isinstance(layer, TFConvTranspose1d) or \
         isinstance(layer, TFResidualStack)


class MelGANGenerator(TFMelGANGenerator):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def set_shape(self, n_mels):
    self.inference = tf.function(self.inference, input_signature=[
      tf.TensorSpec(shape=[None, None, n_mels], dtype=tf.float32, name="mels")
    ])
    self.inference_tflite = tf.function(self.inference_tflite, input_signature=[
      tf.TensorSpec(shape=[1, None, n_mels], dtype=tf.float32, name="mels")
    ])
    self.n_mels = n_mels

  def inference(self, mels):
    return self.melgan(mels)

  def inference_tflite(self, mels):
    return self.melgan(mels)

  def _build(self):
    fake_mels = tf.random.uniform(shape=[1, 100, self.n_mels], dtype=tf.float32)
    self(fake_mels)


class MelGANGeneratorVQ(tf.keras.Model):
  '''VQVAE + MelGAN generator'''
  def __init__(self, config, encoder, **kwargs):
    super().__init__(**kwargs)
    # check hyper parameter is valid or not
    assert config.filters >= np.prod(config.upsample_scales)
    assert config.filters % (2 ** len(config.upsample_scales)) == 0

    # add initial layer
    self.encoder = encoder
    gc_linear = []
    layers = []
    layers += [
        TFReflectionPad1d(
            (config.kernel_size - 1) // 2,
            padding_type=config.padding_type,
            name="first_reflect_padding",
        ),
        tf.keras.layers.Conv1D(
            filters=config.filters,
            kernel_size=config.kernel_size,
            use_bias=config.use_bias,
            kernel_initializer=get_initializer(config.initializer_seed),
        ),
    ]
    gc_linear += [
        tf.keras.layers.Dense(
            units=config.filters,
            kernel_initializer=get_initializer(config.initializer_seed),
            name='gc_start'
        )    
    ]

    for i, upsample_scale in enumerate(config.upsample_scales):
      # add upsampling layer
      layers += [
          getattr(tf.keras.layers, config.nonlinear_activation)(
              **config.nonlinear_activation_params
          ),
          TFConvTranspose1d(
              filters=config.filters // (2 ** (i + 1)),
              kernel_size=upsample_scale * 2,
              strides=upsample_scale,
              padding="same",
              is_weight_norm=config.is_weight_norm,
              initializer_seed=config.initializer_seed,
              name="conv_transpose_._{}".format(i),
          ),
      ]
      gc_linear += [
          tf.keras.layers.Dense(
              units=config.filters // (2 ** (i + 1)),
              activation=tf.nn.tanh,
              kernel_initializer=get_initializer(config.initializer_seed),
              name='gc_%d_0'%(i)
          )    
      ]

      # add residual stack layer
      for j in range(config.stacks):
        layers += [
            TFResidualStack(
                kernel_size=config.stack_kernel_size,
                filters=config.filters // (2 ** (i + 1)),
                dilation_rate=config.stack_kernel_size ** j,
                use_bias=config.use_bias,
                nonlinear_activation=config.nonlinear_activation,
                nonlinear_activation_params=config.nonlinear_activation_params,
                is_weight_norm=config.is_weight_norm,
                initializer_seed=config.initializer_seed,
                name="residual_stack_._{}._._{}".format(i, j),
            )
        ]
        gc_linear += [
            tf.keras.layers.Dense(
                units=config.filters // (2 ** (i + 1)),
                activation=tf.nn.tanh,
                kernel_initializer=get_initializer(config.initializer_seed),
                name='gc_%d_%d'%(i, j+1)
            )    
        ]
    # add final layer
    gc_linear += [
        tf.keras.layers.Dense(
            units=config.out_channels,
            kernel_initializer=get_initializer(config.initializer_seed),
            name='gc_end'
        )    
    ]
    layers += [
        getattr(tf.keras.layers, config.nonlinear_activation)(
            **config.nonlinear_activation_params
        ),
        TFReflectionPad1d(
            (config.kernel_size - 1) // 2,
            padding_type=config.padding_type,
            name="last_reflect_padding",
        ),
        tf.keras.layers.Conv1D(
            filters=config.out_channels,
            kernel_size=config.kernel_size,
            use_bias=config.use_bias,
            kernel_initializer=get_initializer(config.initializer_seed),
            dtype=tf.float32,
        )
    ]
    if config.use_final_nolinear_activation:
      layers += [tf.keras.layers.Activation("tanh", dtype=tf.float32)]

    if config.is_weight_norm is True:
      self._apply_weightnorm(layers)

    self.gc_linear = gc_linear
    self.upsample = layers

  def call(self, mels, gc, training=True, **kwargs):
    """Calculate forward propagation.
    Args:
      mels (Tensor): Input tensor (B, T, channels)
      gc (Tensor): Input tensor (B, 256)
    Returns:
      Tensor: Output tensor (B, T ** prod(upsample_scales), out_channels)
    """
    return self.inference(mels, gc, training)

  def inference(self, mels, gc, training):
    gc = tf.expand_dims(gc, axis=1)
    i = 0
    output = self.encoder(mels, training=training)
    y = output['z_q']
    for layer in self.upsample:
      y = layer(y)
      if if_add_gc(layer):
        y += self.gc_linear[i](gc)
        i += 1
    output.update({'y_mb_hat': y})
    return output

  def inference_tflite(self, mels, gc):
    return self.inference(mels, gc, training=False)['y_mb_hat']

  def _build(self):
    """Build model by passing fake input."""
    fake_mels = tf.random.uniform(shape=[1, 100, self.n_mels], dtype=tf.float32)
    fake_gc = tf.random.uniform(shape=[1, self.gc_channels], dtype=tf.float32)
    self(mels=fake_mels, gc=fake_gc, training=True)

  def set_shape(self, n_mels, gc_channels):
    self.inference = tf.function(self.inference, input_signature=[
      tf.TensorSpec(shape=[None, None, n_mels], dtype=tf.float32, name="mels"),
      tf.TensorSpec(shape=[None, gc_channels], dtype=tf.float32, name="gc"),
      tf.TensorSpec(shape=[], dtype=tf.bool, name="training")    
    ])
    self.inference_tflite = tf.function(self.inference_tflite, input_signature=[
      tf.TensorSpec(shape=[1, None, n_mels], dtype=tf.float32, name="mels"),
      tf.TensorSpec(shape=[1, gc_channels], dtype=tf.float32, name="gc"),
    ])
    self.n_mels = n_mels
    self.gc_channels = gc_channels
