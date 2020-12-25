import numpy as np
import tensorflow as tf
import parameters
from argparse import ArgumentParser
from preprocess import make_asr_model, get_asr_output_from_file

def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--restore', '-r', default='checkpoint')
  parser.add_argument('--src', '-i', default='test.wav')
  args = parser.parse_args()
  return args

def main():

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  args = parse_args()

  asr_model = make_asr_model(args.restore)
  asr_output = get_asr_output_from_file(args.src, asr_model)
  asr, mel, wav = asr_output.values()
  np.save(args.src.replace('.wav', parameters.feature), asr)
  print('finished asr')


if __name__ == '__main__':
  main()