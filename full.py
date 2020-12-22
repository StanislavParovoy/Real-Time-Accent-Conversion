from tensorflow.python.ops.variable_scope import _VARSCOPESTORE_KEY
from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio
from AcousticModel.Conformer.extract_asr_feature import extract
from InverseAcousticModel.melgan import TFMelGANGenerator
from VoiceCloning.GE2E import GE2E
from VoiceCloning.melgan import TFMelGANGeneratorGC

from argparse import ArgumentParser
import numpy as np
from scipy.io import wavfile

def parse_args():
  parser = argparse.ArgumentParser(
      description="Train MultiBand MelGAN (See detail in examples/multiband_melgan/train_multiband_melgan.py)"
  )
  parser.add_argument("--audio", '-a', default='indian.wav')
  return parser.parse_args()

args = parse_args()
audio = np.load(args.audio)
feature, mel = extract(audio)
IAM = TFMelGANGenerator()
IAM.load_weights()
audio_a = IAM(feature, training=False)
wavfile.write('a.wav', 16000, audio_a)
SV = GE2E()
SV.load_weights()
s = SV(mel)
VC = TFMelGANGeneratorGC()
VC.load_weights()
y = VC(audio_a, s)
wavfile.write('y.wav', 16000, y)