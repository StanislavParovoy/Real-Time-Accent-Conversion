# Copyright 2020 Huy Le Nguyen (@usimarit)
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
# import sys
# sys.path.append('.')
import os, yaml
import argparse
from tqdm import tqdm
import numpy as np
from tensorflow_asr.utils import setup_environment, setup_devices
from tensorflow_tts.utils import find_files

import tensorflow as tf

parser = argparse.ArgumentParser(prog="Conformer non streaming")

parser.add_argument("--dataset", '-d', default='../LJSpeech-1.1/wavs')

parser.add_argument("--config", '-c', type=str, default=None,
          help="Path to conformer config yaml")

parser.add_argument("--saved", '-r', type=str, default=None,
          help="Path to conformer saved h5 weights")

args = parser.parse_args()

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer, SubwordFeaturizer
from tensorflow_asr.models.conformer import Conformer

config = Config(args.config, learning=False)
with open(config.speech_config) as f:
  speech_config = yaml.load(f, Loader=yaml.Loader)

with tf.device('/cpu:0'):
  speech_featurizer = TFSpeechFeaturizer(speech_config)
  # build model
  conformer = Conformer(**config.model_config, vocabulary_size=1031)
  conformer._build(speech_featurizer.shape)
  conformer.load_weights(args.saved, by_name=True)
  encoder = conformer.encoder
# encoder.summary(line_length=120)

@tf.function(
  input_signature=[
    tf.TensorSpec(shape=[None], dtype=tf.float32, name="signal")
  ]
)
def extract_from_audio(signal):
  with tf.device('/cpu:0'):
    features = speech_featurizer.tf_extract(signal)
  return extract_from_mel(features)

@tf.function(
  input_signature=[
    tf.TensorSpec(shape=[None, speech_config['n_mels'], 1], dtype=tf.float32, name="signal")
  ]
)
def extract_from_mel(features):
  with tf.device('/cpu:0'):
    encoded = conformer.encoder_inference(features)
  return encoded

suffix = '.wav'
mel_query = '_mel.npy'
feature_query = '_conformer_enc16.npy'
audio_files = sorted(find_files(args.dataset, '*' + suffix))
print('files:', len(audio_files), audio_files[0])

for filename in tqdm(audio_files):
  mel = filename.replace(suffix, mel_query)
  if os.path.exists(mel):
    features = np.load(mel).reshape([-1, speech_config['n_mels'], 1])
    encoded = extract_from_mel(features)
  else:
    signal = read_raw_audio(filename)
    encoded = extract_from_audio(signal)

  np.save(filename.replace(suffix, feature_query), encoded)









