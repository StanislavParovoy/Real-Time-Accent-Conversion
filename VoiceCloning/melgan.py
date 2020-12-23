
import numpy as np
import tensorflow as tf

from tensorflow_tts.utils import GroupConv1D, WeightNormalization
from tensorflow_tts.models.melgan import (get_initializer,
    TFReflectionPad1d, TFConvTranspose1d, TFResidualStack)


n_mels = 80
gc_channels = 256
class TFMelGANGeneratorGC(tf.keras.Model):
    """Tensorflow MelGAN generator module."""
    
    def __init__(self, config, encoder, **kwargs):
        """Initialize TFMelGANGenerator module.
        Args:
            config: config object of Melgan generator.
        """
        super().__init__(**kwargs)

        # check hyper parameter is valid or not
        assert config.filters >= np.prod(config.upsample_scales)
        assert config.filters % (2 ** len(config.upsample_scales)) == 0

        # add initial layer
        self.encoder = encoder
        gc_linear = []
        layers = []
        layers += [tf.keras.Sequential([
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
        ])]
        gc_linear += [
            tf.keras.layers.Dense(
                units=config.filters,
                kernel_initializer=get_initializer(config.initializer_seed),
                name='gc_start'
            )    
        ]

        for i, upsample_scale in enumerate(config.upsample_scales):
            # add upsampling layer
            layers += [tf.keras.Sequential([
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
            ])]
            gc_linear += [
                tf.keras.layers.Dense(
                    units=config.filters // (2 ** (i + 1)),
                    activation=tf.nn.tanh,
                    kernel_initializer=get_initializer(config.initializer_seed),
                    name='gc_%d'%(i)
                )    
            ]

            # ad residual stack layer
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
        # add final layer
        gc_linear += [
            tf.keras.layers.Dense(
                units=config.out_channels,
                kernel_initializer=get_initializer(config.initializer_seed),
                name='gc_end'
            )    
        ]

        layer = [
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
            layer += [tf.keras.layers.Activation("tanh", dtype=tf.float32)]
        layers += [tf.keras.Sequential(layer)]

        if config.is_weight_norm is True:
            self._apply_weightnorm(layers)

        self.gc_linear = gc_linear
        self.upsample = layers
        # self.melgan = tf.keras.models.Sequential(layers)

    def call(self, data, training=True):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, T, channels)
        Returns:
            Tensor: Output tensor (B, T ** prod(upsample_scales), out_channels)
        """
        return self.inference(data, training)

    @tf.function(
        input_signature=[
        {
            'mels': tf.TensorSpec(shape=[None, None, n_mels], dtype=tf.float32, name="mels"),
            'gc': tf.TensorSpec(shape=[None, gc_channels], dtype=tf.float32, name="gc"),
        },
            tf.TensorSpec(shape=[], dtype=tf.bool, name="training")
        ]
    )
    def inference(self, data, training):
        mels, gc = data['mels'], data['gc']
        gc = tf.expand_dims(gc, axis=1)
        i = 0
        output = self.encoder(mels, training=training)
        y = output['z_q']
        for layer in self.upsample:
            y = layer(y)
            if isinstance(layer, tf.keras.Sequential):
                y += self.gc_linear[i](gc)
                i += 1
        output.update({'y_mb_hat': y})
        return output

    @tf.function(
        input_signature=[
        {
            'mels': tf.TensorSpec(shape=[None, None, n_mels], dtype=tf.float32, name="mels"),
            'gc': tf.TensorSpec(shape=[None, gc_channels], dtype=tf.float32, name="gc"),
        }
        ]
    )
    def inference_tflite(self, data):
        mels, gc = data['mels'], data['gc']
        gc = tf.expand_dims(gc, axis=1)
        i = 0
        output = self.encoder(mels, training=False)
        y = output['z_q']
        for layer in self.upsample:
            y = layer(y)
            if isinstance(layer, tf.keras.Sequential):
                y += self.gc_linear[i](gc)
                i += 1
        return y

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass

    def _build(self):
        """Build model by passing fake input."""
        fake_mels = tf.random.uniform(shape=[1, 100, n_mels], dtype=tf.float32)
        fake_gc = tf.random.uniform(shape=[1, gc_channels], dtype=tf.float32)
        data = {'mels': fake_mels, 'gc': fake_gc}
        self(data, training=True)


class TFMelGANGeneratorGCLite(TFMelGANGeneratorGC):
    def __init__(self, pqmf, **kwargs):
        super().__init__(**kwargs)
        self.pqmf = pqmf
    
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, None, n_mels], dtype=tf.float32, name="mels"),
            tf.TensorSpec(shape=[1, gc_channels], dtype=tf.float32, name="gc")
        ]
    )
    def inference_tflite(self, mels, gc):
        gc = tf.expand_dims(gc, axis=1)
        i = 0
        output = self.encoder(mels, training=False)
        y = output['z_q']
        for layer in self.upsample:
            y = layer(y)
            if isinstance(layer, tf.keras.Sequential):
                y += self.gc_linear[i](gc)
                i += 1
        y = self.pqmf.synthesis(y)
        return y
        # return self.melgan(mels, gc)