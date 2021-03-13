# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train Multi-Band MelGAN."""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
  tf.config.experimental.set_memory_growth(physical_devices[i], True)

import sys

sys.path.append(".")
sys.path.append("..")

import argparse
import logging
import os
import numpy as np
import soundfile as sf
import yaml, librosa
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from scipy.io import wavfile

import tensorflow_tts

from tensorflow_tts.configs import (
  MultiBandMelGANGeneratorConfig,
)

from tensorflow_tts.models import (
  TFPQMF,
)
from tensorflow_tts.utils import return_strategy
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer

from rtac.models import Encoder, MelGANGeneratorVQ

def main():
  """Run training process."""
  parser = argparse.ArgumentParser(
    description="Train MultiBand MelGAN (See detail in examples/multiband_melgan/train_multiband_melgan.py)"
  )
  parser.add_argument("--feature", '-f', required=True)
  parser.add_argument("--speaker", '-s', required=True)
  parser.add_argument("--config", '-c', required=True)
  parser.add_argument("--resume", '-r', required=True)
  args = parser.parse_args()

  # return strategy
  STRATEGY = return_strategy()

  # load and save config
  with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)
  with open(config['speech_config']) as f:
    speech_config = yaml.load(f, Loader=yaml.Loader)
  config.update(speech_config)
  config['hop_size'] = config['sample_rate'] * config['stride_ms'] // 1000
  config['sampling_rate'] = config['sample_rate']

  config.update(vars(args))
  config["version"] = tensorflow_tts.__version__
  for key, value in config.items():
    logging.info(f"{key} = {value}")

  with STRATEGY.scope():
    encoder = Encoder(**config['encoder'])

    generator = MelGANGeneratorVQ(
      encoder=encoder,
      config=MultiBandMelGANGeneratorConfig(
      **config["multiband_melgan_generator_params"]
      ),
      name="multi_band_melgan_generator",
    )
    generator.set_shape(config['n_mels'], config['gc_channels'])

    pqmf = TFPQMF(
      MultiBandMelGANGeneratorConfig(
        **config["multiband_melgan_generator_params"]
      ),
      dtype=tf.float32,
      name="pqmf",
    )

    # dummy input to build model.
    fake_mels = tf.random.uniform(shape=[1, 100, config['n_mels']], dtype=tf.float32)
    fake_gc = tf.random.uniform(shape=[1, config['gc_channels']], dtype=tf.float32)
    y_mb_hat = generator(mels=fake_mels, gc=fake_gc, training=False)['y_mb_hat']
    y_hat = pqmf.synthesis(y_mb_hat)

    generator.load_weights(args.resume)
    generator.summary()

  speech_featurizer = TFSpeechFeaturizer(speech_config)
  if args.feature.endswith('_mel.npy'):
    mels = tf.constant(np.load(args.feature), tf.float32)
  else:
    signal, _ = librosa.load(args.feature, sr=config['sample_rate'])
    mels = speech_featurizer.tf_extract(signal)
  mels = tf.reshape(mels, [1, -1, config['n_mels']])

  gc = tf.constant(np.load(args.speaker).reshape([1, config['gc_channels']]), tf.float32)
  # gc = tf.constant(np.zeros(256).reshape([1, config['gc_channels']]), tf.float32)
  output = generator(mels=mels, gc=gc, training=False)['y_mb_hat']
  y_hat = pqmf.synthesis(output).numpy().reshape([-1])
  print('output:', y_hat.shape)
  save_name = args.feature.replace('.wav', '_gen_vc.wav')
  save_name = args.feature.replace('_mel.npy', '_gen_vc.wav')
  save_name = save_name.split('/')[-1]
  wavfile.write(save_name, config['sample_rate'], y_hat)

  def depreemphasis(signal: np.ndarray, coeff=0.97):
    if not coeff or coeff <= 0.0: return signal
    x = np.zeros(signal.shape[0], dtype=np.float32)
    x[0] = signal[0]
    for n in range(1, signal.shape[0], 1):
      x[n] = coeff * x[n - 1] + signal[n]
    return x
  y_hat = depreemphasis(y_hat)
  wavfile.write(save_name.replace('.wav', '_depre.wav'), config['sample_rate'], y_hat)

if __name__ == "__main__":
  main()
