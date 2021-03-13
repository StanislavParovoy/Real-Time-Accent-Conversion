import os, yaml, sys, time
import argparse
from tqdm import tqdm
import numpy as np
import librosa
from scipy.io import wavfile
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow_tts.utils import find_files
from tensorflow_tts.configs import MultiBandMelGANGeneratorConfig
from tensorflow_tts.models import TFPQMF

from tensorflow_asr.utils import setup_environment, setup_devices
from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio, TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer, SubwordFeaturizer
from tensorflow_asr.models.conformer import Conformer

sys.path.append('.')
from rtac.models import MelGANGenerator, GE2E, Encoder, MelGANGeneratorVQ

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', '-i', help='path to source audio')
  parser.add_argument('--trim-silence', '-t', action='store_true', help='whether to trim trailing silence')

  parser.add_argument('--am-config', '-amc', default='acoustic_model/subword_conformer.yml')
  parser.add_argument('--am-model', '-am', help='path to acoustic model weight')

  parser.add_argument('--iam-config', '-iamc', default='inverse_acoustic_model/iam_conformer.yaml')
  parser.add_argument('--iam-model', '-iam', help='path to inverse acoustic model weight')

  parser.add_argument('--sv-config', '-svc', default='ge2e/ge2e.yml')
  parser.add_argument('--sv-model', '-sv', help='path to speaker verification weight')

  parser.add_argument('--vc-config', '-vcc', default='voice_cloning/vc.yaml')
  parser.add_argument('--vc-model', '-vc', help='path to voice cloning weight')
  args = parser.parse_args()
  return args


class Inference(tf.keras.Model):
  def __init__(self, args):
    super().__init__()
    with open(args.am_config) as f:
      am_config = yaml.load(f, Loader=yaml.Loader)

    with open(am_config['speech_config']) as f:
      self.speech_config = yaml.load(f, Loader=yaml.Loader)
    self.speech_featurizer = TFSpeechFeaturizer(self.speech_config)

    self.am = self.build_am(args.am_config, args.am_model)

    with open(args.iam_config) as f:
      iam_config = yaml.load(f, Loader=yaml.Loader)
    iam_config.update(self.speech_config)
    iam_config['n_mels'] = iam_config['asr_features']
    iam_config['hop_size'] = iam_config['asr_downsample'] * iam_config['sample_rate'] * iam_config['stride_ms'] // 1000

    self.iam, self.pqmf = self.build_iam(iam_config, args.iam_model)

    with open(args.sv_config) as f:
      sv_config = yaml.load(f, Loader=yaml.Loader)
    sv_config.update(self.speech_config)

    self.sv = self.build_sv(sv_config, args.sv_model)

    with open(args.vc_config) as f:
      vc_config = yaml.load(f, Loader=yaml.Loader)
    vc_config.update(self.speech_config)
    vc_config['hop_size'] = vc_config['sample_rate'] * vc_config['stride_ms'] // 1000
    vc_config['sampling_rate'] = vc_config['sample_rate']

    self.vc = self.build_vc(vc_config, args.vc_model)

  # @tf.function
  def call(self, x):
    c = self.speech_featurizer.tf_extract(x)
    gc = self.sv(tf.reshape(c, [1, -1, self.speech_config['n_mels']]))

    with tf.device('/cpu:0'):
      c = self.am.encoder_inference(c)
      c = tf.expand_dims(c, 0)

    x = self.iam(mels=c, training=False)
    x = self.pqmf.synthesis(x)
    x = tf.squeeze(x)

    c = self.speech_featurizer.tf_extract(x)
    c = tf.reshape(c, [1, -1, self.speech_config['n_mels']])

    y = self.vc(mels=c, gc=gc, training=False)['y_mb_hat']
    y = self.pqmf.synthesis(y)
    y = tf.squeeze(y)
    
    return x, y

  def build_am(self, config_path, model_path):
    config = Config(config_path, learning=False)
    conformer = Conformer(**config.model_config, vocabulary_size=1031)
    conformer._build(self.speech_featurizer.shape)
    print('loading am...')
    conformer.load_weights(model_path, by_name=True)
    return conformer

  def build_iam(self, config, model_path):
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
    fake_mels = tf.random.uniform(shape=[1, 100, config['n_mels']], dtype=tf.float32)
    output = generator(mels=fake_mels, training=False)
    y_hat = pqmf.synthesis(output)
    print('loading iam...')
    generator.load_weights(model_path)
    return generator, pqmf

  def build_sv(self, config, model_path):
    model = GE2E(name='ge2e', **config['model'])
    fake_mels = tf.random.uniform(shape=[1, 100, config['n_mels']], dtype=tf.float32)
    model(fake_mels)
    print('loading sv...')
    model.load_weights(model_path)
    return model

  def build_vc(self, config, model_path):
    encoder = Encoder(**config['encoder'])
    generator = MelGANGeneratorVQ(
      encoder=encoder,
      config=MultiBandMelGANGeneratorConfig(
        **config["multiband_melgan_generator_params"]
      ),
      name="multi_band_melgan_generator",
    )
    generator.set_shape(config['n_mels'], config['gc_channels'])

    fake_mels = tf.random.uniform(shape=[1, 100, config['n_mels']], dtype=tf.float32)
    fake_gc = tf.random.uniform(shape=[1, config['gc_channels']], dtype=tf.float32)
    y_mb_hat = generator(mels=fake_mels, gc=fake_gc, training=False)['y_mb_hat']
    print('loading vc...')
    generator.load_weights(model_path)
    return generator

def main():
  args = parse_args()
  model = Inference(args)

  x, sr = librosa.load(args.source, sr=model.speech_config['sample_rate'])
  if args.trim_silence:
    x, _ = librosa.effects.trim(x, 
        top_db=30, 
        frame_length=2048, 
        hop_length=512)

  t = time.time()
  x, y = model(x)

  x, y = x.numpy(), y.numpy()
  t = time.time() - t
  print('generation time: %.4fs, rtf: %.4f' % (t, t / (y.shape[0] / sr)))

  name = args.source.split('/')[-1].replace('.wav', '')
  wavfile.write(name + '_iam.wav', sr, x)
  wavfile.write(name + '_vc.wav', sr, y)

if __name__ == '__main__':
  main()

