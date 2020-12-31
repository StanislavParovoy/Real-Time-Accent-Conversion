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
# import sys
# sys.path.append('.')
import os, yaml
import argparse
from tensorflow_asr.utils import setup_environment, setup_devices
from tensorflow_tts.utils import find_files

setup_environment()
import tensorflow as tf

parser = argparse.ArgumentParser(prog="Conformer non streaming")

parser.add_argument("--dataset", '-d', default='../LJSpeech-1.1/wavs')

parser.add_argument("--config", type=str, default=None,
                    help="Path to conformer config yaml")

parser.add_argument("--saved", type=str, default=None,
                    help="Path to conformer saved h5 weights")

parser.add_argument("--device", type=int, default=0,
                    help="Device's id to run test on")

parser.add_argument("--cpu", default=False, action="store_true",
                    help="Whether to only use cpu")

parser.add_argument("--output_name", type=str, default="test",
                    help="Result filename name prefix")

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
def extract(signal):
    features = speech_featurizer.tf_extract(signal)
    encoded = conformer.encoder_inference(features)
    return encoded

suffix = '.wav'
audio_query = '_raw.npy'
feature_query = '_conformer_enc16.npy'
audio_files = sorted(find_files(args.dataset, '*' + suffix))
print('files:', len(audio_files), audio_files[0])
from tqdm import tqdm
import numpy as np
for filename in tqdm(audio_files):
    signal = read_raw_audio(filename)
    encoded = extract(signal)
    np.save(filename.replace(suffix, audio_query), signal)
    np.save(filename.replace(suffix, feature_query), encoded)









