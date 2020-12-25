from pathlib import Path
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import os, shutil
import librosa

import tensorflow as tf
from tdnn_encoder import TDNNEncoder
from jasper10x5_LibriSpeech_nvgrad import base_params
from speech_utils import get_speech_features_from_file
from argparse import ArgumentParser


def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--dataset', '-d', default='../VCTK-Corpus/wav48')
  parser.add_argument('--checkpoint', '-c', default='checkpoint')
  parser.add_argument('--asr', '-a', default=False, action='store_true')
  args = parser.parse_args()
  return args


def mulaw_encode(x, mu):
  mu = mu - 1
  fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
  return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
  y = y * 2 / mu - 1
  mu = mu - 1
  x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
  return x


def compute_f0(y, fmin, fmax, frame_length, win_length, hop_length):
  f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, fill_na=fmin,
    frame_length=frame_length, win_length=win_length, hop_length=hop_length)
  f0 = f0.astype(np.float32)
  f0 = np.log(f0)
  f0 = np.expand_dims(f0, -1)
  return f0

def get_asr_output_from_file(filename, min_len=100, even=True,
    model=None, mel=None, wav=None):
  # mel, wav
  if mel is None:
    mel, _, wav = get_speech_features_from_file(filename, base_params['data_layer_params'], return_wav=True)
  if even and mel.shape[0] % 2 == 1:
    mel = mel[:-1]
  if min_len is not None and mel.shape[0] < min_len:
    return None
  # f0
  p = base_params['data_layer_params']
  f0 = compute_f0(wav, fmin=100, fmax=2093,
    frame_length=1024,
    win_length=int(p['window_size']*p['sample_freq']), 
    hop_length=int(p['window_stride']*p['sample_freq']))
  f0 = f0[:mel.shape[0]]
  # asr feature
  if model is None:
    y105, y115 = None, None
  else:
    y105, y115 = model['sess'].run([model['y105'], model['y115']], {model['x']: np.expand_dims(mel, 0)})
  return {'y105': y105, 'y115': y115, 'mel': mel, 'wav': wav, 'f0': f0}


def make_asr_model(checkpoint):
  x = tf.compat.v1.placeholder(shape=[1, None, 64], dtype=tf.float32)
  input_dict = {"source_tensors": [x, None]}

  params = base_params['encoder_params']

  model = TDNNEncoder(params, None, 'ForwardPass/w2l_encoder', 'infer')
  y105, y115, _ = model.encode(input_dict).values()
  if y115.shape.as_list()[-1] == 1024:
    name = 'ForwardPass/fully_connected_ctc_decoder/fully_connected'
    layer = tf.keras.layers.Dense(29, name=name)
    y115 = layer(y115)

  sess = tf.compat.v1.Session()
  latest_checkpoint = tf.train.latest_checkpoint(checkpoint)

  def restore(sess, scope, checkpoint):
    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint)
    variables = tf.compat.v1.global_variables(scope=scope)
    assign_ops = []
    # for var in variables:
    print('restoring asr model')
    for var in tqdm(variables):
      try:
        idx = var.name.find(":")
        if idx != -1:
          true_name = var.name[:idx]
        else:
          true_name = var.name
        tensor = reader.get_tensor(true_name)
        if tensor.dtype != var.dtype.as_numpy_dtype():
          assign_ops.append(var.assign(tf.cast(tensor, var.dtype)))
        else:
          assign_ops.append(var.assign(tensor))
      except Exception as e:
        print('FAILED', var.name, var.shape, var.dtype, e)
    sess.run(assign_ops)
    print('finished restoration')

  restore(sess, 'ForwardPass', latest_checkpoint)
  return {'x': x, 'y105': y105, 'y115': y115, 'sess': sess}


def process_file(path, bits=8, model=None):
  mel = np.load(path)
  wav = np.load(path.replace('_mel64.npy', '_wav.npy'))
  asr_output = get_asr_output_from_file(path.replace('_mel64.npy', '.wav'), 
      min_len=100, even=True, model=None, mel=mel, wav=wav)
  if asr_output is None:
    return
  y105, y115, mel, wav, f0 = asr_output.values()
  np.save(path.replace('_mel64.npy', '_f0.npy'), f0)
  return



  out_path = path.replace('.wav', '')
  try:
    if model is not None:
      # mel = np.load(path.replace('.wav', '_mel64.npy'))
      mel = np.load(path)
      wav = np.load(path.replace('_mel64.npy', '_wav.npy'))
      asr_output = get_asr_output_from_file(path.replace('_mel64.npy', '.wav'), 
        min_len=100, even=True, model=model, mel=mel, wav=wav)
    else:
      asr_output = get_asr_output_from_file(path, min_len=100, even=True)
    if asr_output is None:
      return

    y105, y115, mel, wav, f0 = asr_output.values()
    if model is not None:
      np.save(path.replace('_mel64.npy', '_asr_conv105.npy'), y105[0])
      np.save(path.replace('_mel64.npy', '_asr_conv115.npy'), y115[0])
    else:
      wav_mu = mulaw_encode(wav, mu=2**bits)
      all_suffix = ['_wav', '_wav_mu', '_mel64', '_f0']
      for name, obj in zip(all_suffix, [wav, wav_mu, mel, f0]):
        np.save(out_path + name + '.npy', obj)
  except Exception as e:
    print('exception:', e)
    exit()
    for f in all_suffix:
      try:
        os.remove(out_path + f + '.npy')
      except:
        pass

def preprocess_dataset(args):
    executor = ProcessPoolExecutor(max_workers=cpu_count())
    futures = []

    root = Path(args.dataset)
    if not args.asr:
      suffix = '.wav'
    else:
      suffix = '_mel64.npy'

    depth = 2
    if 'lj' in args.dataset.lower():
      depth = 1
    elif 'vctk' in args.dataset.lower():
      depth = 2
    elif 'libri' in args.dataset.lower():
      depth = 3
    pattern = '/'.join('*'*depth) + suffix

    all_filenames = root.glob(pattern)
    all_filenames = [str(f) for f in all_filenames]
    print('num files total: %d' % len(all_filenames), all_filenames[0])

    if not args.asr:
      for file in all_filenames:
        futures.append(executor.submit(partial(process_file, file, 8)))
      results = [future.result() for future in tqdm(futures)]
    else:
      model = make_asr_model(args.checkpoint)
      for file in tqdm(all_filenames):
        process_file(file, 8, model)


if __name__ == "__main__":
  tf.compat.v1.disable_v2_behavior()
  args = parse_args()
  preprocess_dataset(args)

