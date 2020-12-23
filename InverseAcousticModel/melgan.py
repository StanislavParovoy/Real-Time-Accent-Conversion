
import numpy as np
import tensorflow as tf

from tensorflow_tts.utils import GroupConv1D, WeightNormalization
from tensorflow_tts.models.melgan import (get_initializer,
    TFReflectionPad1d, TFConvTranspose1d, TFResidualStack)


n_mels = 768
class TFMelGANGenerator(tf.keras.Model):
    """Tensorflow MelGAN generator module."""

    def __init__(self, config, **kwargs):
        """Initialize TFMelGANGenerator module.
        Args:
            config: config object of Melgan generator.
        """
        super().__init__(**kwargs)

        # check hyper parameter is valid or not
        assert config.filters >= np.prod(config.upsample_scales)
        assert config.filters % (2 ** len(config.upsample_scales)) == 0

        # add initial layer
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
            ),
        ]
        if config.use_final_nolinear_activation:
            layers += [tf.keras.layers.Activation("tanh", dtype=tf.float32)]

        if config.is_weight_norm is True:
            self._apply_weightnorm(layers)

        self.melgan = tf.keras.models.Sequential(layers)

    def call(self, mels, **kwargs):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, T, channels)
        Returns:
            Tensor: Output tensor (B, T ** prod(upsample_scales), out_channels)
        """
        return self.inference(mels)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, n_mels], dtype=tf.float32, name="mels")
        ]
    )
    def inference(self, mels):
        return self.melgan(mels)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, None, n_mels], dtype=tf.float32, name="mels")
        ]
    )
    def inference_tflite(self, mels):
        return self.melgan(mels)

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
        self(fake_mels)

