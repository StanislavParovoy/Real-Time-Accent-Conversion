
import argparse
import yaml
import sys
sys.path.append(".")
sys.path.append("..")

import tensorflow as tf

from VoiceCloning.vq import Encoder
from Melgan.melgan import TFMelGANGeneratorGC
from tensorflow_tts.models import TFPQMF
from tensorflow_tts.configs import MultiBandMelGANGeneratorConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", '-c',
    default=None,
    type=str,
)
parser.add_argument(
    "--restore", '-r',
    default=None,
    type=str,
)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)
config.update(vars(args))

encoder = Encoder()
generator = TFMelGANGeneratorGC(
    config=MultiBandMelGANGeneratorConfig(
        **config["multiband_melgan_generator_params"]
    ),
    encoder=encoder,
    name="multi_band_melgan_generator",
)
pqmf = TFPQMF(
    MultiBandMelGANGeneratorConfig(
        **config["multiband_melgan_generator_params"]
    ),
    dtype=tf.float32,
    name="pqmf",
)

class Model(tf.keras.Model):
  def __init__(self, generator, pqmf, **kwargs):
    super().__init__(**kwargs)
    generator._build()
    self.generator = generator
    self.pqmf = pqmf
    
  @tf.function(
      input_signature=[{
          'mels': tf.TensorSpec(shape=[1, None, config['n_mels']], dtype=tf.float32, name="mels"),
          'gc': tf.TensorSpec(shape=[1, config['gc_channels']], dtype=tf.float32, name="gc")
      }]
  )
  def call(self, data):
    x = self.generator.inference_tflite(data)
    x = self.pqmf.synthesis(x)
    return x

model = Model(generator, pqmf)
fake_mels = tf.random.uniform(shape=[1, 100, config['n_mels']], dtype=tf.float32)
fake_gc = tf.random.uniform(shape=[1, config['gc_channels']], dtype=tf.float32)
y = model({'mels': fake_mels, 'gc': fake_gc})
print('y:', y)
# y = model.inference_tflite(fake_mels, fake_gc)
# print('y:', y)
converter = tf.lite.TFLiteConverter.from_keras_model(generator)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)


