# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, yaml
import argparse
from tensorflow_asr.utils import setup_environment, setup_devices
from tensorflow_asr.utils.utils import get_reduced_length

setup_environment()
import tensorflow as tf

parser = argparse.ArgumentParser(prog="Conformer non streaming")

parser.add_argument("--filename", '-f', metavar="FILENAME", help="audio file to be played back")

parser.add_argument("--config", '-c', type=str, default=None, help="Path to conformer config yaml")

parser.add_argument("--saved", '-r', type=str, default=None, help="Path to conformer saved h5 weights")

parser.add_argument("--beam_width", '-b', type=int, default=0, help="Beam width")

parser.add_argument("--timestamp", '-t', default=False, action="store_true", help="Return with timestamp")

parser.add_argument("--device", '-gpu', type=int, default=0, help="Device's id to run test on")

parser.add_argument("--cpu", '-cpu', default=False, action="store_true", help="Whether to only use cpu")

parser.add_argument("--subwords", '-sub', type=str, default=None, help="Path to file that stores generated subwords")

args = parser.parse_args()

setup_devices([args.device], cpu=args.cpu)

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer, SubwordFeaturizer
from tensorflow_asr.models.conformer import Conformer

config = Config(args.config, learning=False)
with open(config.speech_config) as f:
  speech_config = yaml.load(f, Loader=yaml.Loader)

speech_featurizer = TFSpeechFeaturizer(speech_config)
if args.subwords and os.path.exists(args.subwords):
  print("Loading subwords ...")
  text_featurizer = SubwordFeaturizer.load_from_file(config.decoder_config, args.subwords)
else:
  text_featurizer = CharFeaturizer(config.decoder_config)
text_featurizer.decoder_config.beam_width = args.beam_width

# build model
conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
conformer._build(speech_featurizer.shape)
conformer.load_weights(args.saved, by_name=True)
conformer.summary(line_length=120)
conformer.add_featurizers(speech_featurizer, text_featurizer)

import numpy as np
np.random.seed(0)
tf.random.set_seed(0)
if args.filename.endswith('.wav'):
  signal = read_raw_audio(args.filename)
  # features = speech_featurizer.tf_extract(signal)
  features = speech_featurizer.extract(signal)
  features = tf.constant(features)
else:
  features = np.load(args.filename).reshape([-1, 80, 1])
  features = tf.constant(features)
input_length = get_reduced_length(tf.shape(features)[0], conformer.time_reduction_factor)

if args.beam_width:
  transcript = conformer.recognize_beam(features[None, ...], input_length[None, ...])
  print("Transcript:", transcript[0].numpy().decode("UTF-8"))
elif args.timestamp:
  transcript, stime, etime, _, _ = conformer.recognize_tflite_with_timestamp(
    signal, tf.constant(text_featurizer.blank, dtype=tf.int32), conformer.predict_net.get_initial_state())
  print("Transcript:", transcript)
  print("Start time:", stime)
  print("End time:", etime)
else:
  if args.filename.endswith('.wav'):
    transcript, _, _ = conformer.recognize_tflite(
      signal, tf.constant(text_featurizer.blank, dtype=tf.int32), conformer.predict_net.get_initial_state())
    print("Transcript:", tf.strings.unicode_encode(transcript, "UTF-8").numpy().decode("UTF-8"))
  else:
    encoded = conformer.encoder_inference(features)
    hypothesis = conformer._perform_greedy(
        encoded, 
        tf.shape(encoded)[0], 
        tf.constant(text_featurizer.blank, dtype=tf.int32), 
        conformer.predict_net.get_initial_state())
    transcript = conformer.text_featurizer.indices2upoints(hypothesis.prediction)
    print("Transcript:", tf.strings.unicode_encode(transcript, "UTF-8").numpy().decode("UTF-8"))





