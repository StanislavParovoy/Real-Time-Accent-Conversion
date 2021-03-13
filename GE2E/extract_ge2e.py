from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import json, time, io, soundfile, os, yaml, sys

sys.path.append(".")
from rtac.models import GE2E
from tensorflow_tts.utils import find_files

def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--config', '-c', default='ge2e/ge2e.yml')
  parser.add_argument('--dataset', '-d', 
      nargs='*',
      default=['../LibriTTS/train-other-500'])
  parser.add_argument('--n_gpu', '-g', type=int, default=1)
  parser.add_argument('--restore', '-r', default='saved_ge2e')
  args = parser.parse_args()
  return args

def batch_frames(frames, window_length, overlap=0.5):
  ''' (for inference) batch spectrogram a long audio s.t. shape[1] = window_length
  Args:
    frames: np array, shape [length, n_mels]
    window_length: int
    overlap: ~(0, 1], will be multiplied by window_length
  Returns:
    batched frames, np array, shape [?, 2 * min_frames]
  '''
  overlap = int(window_length * overlap)
  num_windows = np.shape(frames)[0] // overlap
  frames = frames[: num_windows * overlap]

  frames = [frames[i * overlap: i * overlap + window_length] for i in range(num_windows - 1)]
  frames = np.reshape(frames, [num_windows - 1, window_length, -1])
  return frames

def main():
  args = parse_args()
  args.train_dataset = args.eval_dataset = args.dataset
  with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)
  with open(config['speech_config']) as f:
    mel_config = yaml.load(f, Loader=yaml.Loader)
  config.update(mel_config)
  config.update(vars(args))

  model = GE2E(name='ge2e', **config['model'])
  model.load_weights(args.restore)

  window_length = (config['data']['min_frames'] + config['data']['max_frames']) // 2

  suffix = '*_mel.npy'
  save_as = '_gc.npy'
  for dataset in args.dataset:
    files = find_files(dataset, suffix)
    print('files of %s:'% dataset, len(files), files[0])

    for file in tqdm(files):
      save_name = file.replace(suffix[1:], save_as)
      # if os.path.isfile(save_name):
        # continue
      s = np.load(file)
      len_s = len(s)
      if len_s < window_length:
        s = np.concatenate([s for _ in range((window_length + len_s) // len_s)])
      s = batch_frames(s, window_length=window_length, overlap=0.5)

      d = model.inference(s)
      np.save(save_name, d)

if __name__ == '__main__':
  main()

