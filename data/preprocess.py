from pathlib import Path
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from argparse import ArgumentParser
from tqdm import tqdm
import os, shutil, yaml
import librosa

import tensorflow as tf
from tensorflow_asr.featurizers.speech_featurizers import (
  TFSpeechFeaturizer,
  normalize_signal,
  preemphasis
)
from tensorflow_tts.utils import find_files

def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--dataset', '-d', required=True)
  parser.add_argument('--suffix', '-s', default='*.wav')
  parser.add_argument('--config', '-c', default='data/config.yml')
  args = parser.parse_args()
  return args

def compute_f0(y, fmin, fmax, frame_length, win_length, hop_length):
  f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, fill_na=fmin,
    frame_length=frame_length, win_length=win_length, hop_length=hop_length)
  f0 = f0.astype(np.float32)
  f0 = np.log(f0)
  f0 = np.expand_dims(f0, -1)
  return f0

def process_file(path, model, suffix):
  try:
    os.remove(path.replace(suffix, '_mel64.npy'))
  except:
    pass
  try:
    if model is not None:
      wav, _ = librosa.load(path, sr=model.sample_rate)
      with tf.device("/CPU:0"):
        mel = model.extract(wav)
      wav = np.asfortranarray(wav)
      if model.normalize_signal:
          wav = normalize_signal(wav)
      wav = preemphasis(wav, model.preemphasis)
    else:
      return

    np.save(path.replace(suffix, '_wav.npy'), wav)
    np.save(path.replace(suffix, '_mel.npy'), mel)

  except Exception as e:
    raise e

def main():

  args = parse_args()

  with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)
    config = config['speech_config']

  model = TFSpeechFeaturizer(config)
  executor = ProcessPoolExecutor(max_workers=cpu_count())

  all_filenames = find_files(args.dataset, args.suffix)

  futures = []

  print('num files total: %d' % len(all_filenames), all_filenames[0])

  suffix = args.suffix.replace('*', '')
  # for file in all_filenames:
  #   futures.append(executor.submit(partial(process_file, file, model, suffix)))
  # results = [future.result() for future in tqdm(futures)]

  for file in tqdm(all_filenames):
    process_file(file, model, suffix)


if __name__ == "__main__":
  main()

