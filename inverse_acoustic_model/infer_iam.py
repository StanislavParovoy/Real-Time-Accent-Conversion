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
import yaml
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from scipy.io import wavfile

import tensorflow_tts

from tensorflow_tts.configs import (
  MultiBandMelGANDiscriminatorConfig,
  MultiBandMelGANGeneratorConfig,
)

from tensorflow_tts.models import (
  TFPQMF,
  TFMelGANMultiScaleDiscriminator,
)
from tensorflow_tts.utils import return_strategy

from rtac.models import MelGANGenerator


def main():
  """Run training process."""
  parser = argparse.ArgumentParser(
    description="Train MultiBand MelGAN (See detail in examples/multiband_melgan/train_multiband_melgan.py)"
  )
  parser.add_argument("--feature", '-f', required=True)
  parser.add_argument("--config", '-c', required=True)
  parser.add_argument("--restore_am", '-am', required=True)
  parser.add_argument("--restore_iam", '-iam', required=True)
  args = parser.parse_args()

  # return strategy
  STRATEGY = return_strategy()

  # load and save config
  with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)
  with open(config['speech_config']) as f:
    speech_config = yaml.load(f, Loader=yaml.Loader)
  config.update(speech_config)
  config['n_mels'] = config['asr_features']
  config['hop_size'] = config['asr_downsample'] * config['sample_rate'] * config['stride_ms'] // 1000

  config.update(vars(args))
  config["version"] = tensorflow_tts.__version__
  for key, value in config.items():
    logging.info(f"{key} = {value}")

  with STRATEGY.scope():
    generator = MelGANGenerator(
      config=MultiBandMelGANGeneratorConfig(
        **config["multiband_melgan_generator_params"]
      ),
      name="multi_band_melgan_generator",
    )
    generator.set_shape(config['n_mels'])

    pqmf = TFPQMF(
      MultiBandMelGANGeneratorConfig(
        **config["multiband_melgan_generator_params"]
      ),
      dtype=tf.float32,
      name="pqmf",
    )

    # dummy input to build model.
    fake_mels = tf.random.uniform(shape=[1, 100, config['n_mels']], dtype=tf.float32)
    output = generator(mels=fake_mels, training=False)
    y_hat = pqmf.synthesis(output)
    print('y_hat', y_hat.shape)
    generator.load_weights(args.resume)


  elif args.feature.endswith('.wav'):
    signal, _ = librosa.load(args.feature, sr=config['sample_rate'])
    mels = speech_featurizer.tf_extract(signal)
    with tf.device('/cpu:0'):
      mels = conformer.encoder_inference(mels)


  mels = tf.reshape(mels, [1, -1, config['n_mels']])

  # mels = tf.constant(np.load(args.feature).reshape([1, -1, config['n_mels']]), tf.float32)
  output = generator(mels=mels, training=False)
  y_hat = pqmf.synthesis(output).numpy().reshape([-1])
  print('output:', y_hat.shape)
  save_name = args.feature.replace('_conformer_enc16.npy', '.wav')
  save_name = save_name.split('/')[-1]
  wavfile.write(save_name, config['sample_rate'], y_hat)

if __name__ == "__main__":
  main()
