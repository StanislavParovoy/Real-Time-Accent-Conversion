import parameters
from model import GE2E
from preprocess import get_file_of_speaker
from utils import *

from argparse import ArgumentParser
from tqdm import tqdm
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np
import json, time, io, soundfile, os

def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--restore', '-r')
  parser.add_argument('--dataset', '-d', default='../../data/VCTK-Corpus/wav48')
  args = parser.parse_args()
  return args

def batch_frames(frames, window_length, overlap=0.5):
  ''' (for inference) batch spectrogram a long audio s.t. shape[1] = window_length
  args:
    frames: np array, shape [length, n_mels]
    window_length: int
    overlap: ~(0, 1], will be multiplied by window_length
  returns:
    batched frames, np array, shape [?, 2 * min_frames]
  '''
  overlap = int(window_length * overlap)
  num_windows = np.shape(frames)[0] // overlap
  frames = frames[: num_windows * overlap]

  frames = [frames[i * overlap: i * overlap + window_length] for i in range(num_windows - 1)]
  frames = np.reshape(frames, [num_windows - 1, window_length, -1])
  return frames

def infer(sess, frames, window_length, embedding, window):
  y = batch_frames(frames, window_length=window_length, overlap=0.5)
  d_vector = sess.run(embedding, {window: y})
  return d_vector

def main():
  args = parse_args()

  window_length = (parameters.min_frames + parameters.max_frames) // 2

  window = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, window_length, parameters.n_mels])
  model = GE2E(scope='ge2e', training=False)
  embedding = model(window)

  global_step = tf.Variable(tf.constant(0, dtype=tf.int32), trainable=False, name='global_step')

  sess = tf.compat.v1.Session()
  if parameters.ema:
    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    variables = ema.variables_to_restore()
    saver = tf.compat.v1.train.Saver(variables)
  else:
    var_list = {}
    for v in tf.compat.v1.global_variables():
      var_list[v.name.replace('lstm_cell/', '').replace(':0', '')] = v
    saver = tf.compat.v1.train.Saver(var_list)
  latest_checkpoint = tf.train.latest_checkpoint(args.restore)
  if latest_checkpoint is not None:
    print('restore from', latest_checkpoint)
    saver.restore(sess, latest_checkpoint)
  else:
    print('failed to rstore')
    exit()
    # sess.run(tf.compat.v1.global_variables_initializer())
  gs = sess.run(global_step)

  if 'libri' in args.dataset.lower():
    depth = 3
  elif 'vctk' in args.dataset.lower():
    depth = 2
  else:
    depth = 2
  base = '_mel64.npy'
  file_of_speaker = get_file_of_speaker(args.dataset, depth, base)
  file_and_speaker = [(speaker, file) for speaker, files in file_of_speaker.items() for file in files]

  for speaker, file in tqdm(file_and_speaker):
    save_name = file.replace(base, '_gc.npy')
    if os.path.isfile(save_name):
      continue
    s = np.load(file)
    len_s = len(s)
    if len_s < window_length:
      s = np.concatenate([s for _ in range((window_length + len_s) // len_s)])

    d = infer(sess, s, window_length, embedding, window)
    np.save(save_name, d)

if __name__ == '__main__':
  suppress_tf_warning()
  main()

