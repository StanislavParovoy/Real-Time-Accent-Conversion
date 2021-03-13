
import csv
import numpy as np
from pathlib import Path
from tensorflow_tts.utils import find_files

def create_libritts(root, save_name):
  lines = []
  txt = '.normalized.txt'
  print(f"Reading {root} ...")
  all_files = map(str, Path(root).glob('*/*/*'+txt))
  for filename in all_files:
    with open(filename) as f:
      text = f.readline()
    audio_name = filename.replace(txt, '.wav')
    lines.append((audio_name, text))

  lines = sorted(lines, key=lambda t: t[0])

  with open('libritts_%s.tsv'%save_name, 'w') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['audio_name', 'transcript'])
    for line in lines:
      writer.writerow(line)

  return lines

def create_vctk(root, save_name, split):
  lines = []
  txt = '.txt'
  print(f"Reading {root} ...")
  all_files = map(str, Path(root.replace('wav48', 'txt')).glob('*/*'+txt))
  for filename in all_files:
    with open(filename) as f:
      text = f.readline().strip()
    audio_name = filename.replace(txt, '.wav').replace('txt', 'wav48')
    lines += [(audio_name, text)]
  # lines = sorted(lines, key=lambda t: t[0])
  lines = np.array(lines)
  np.random.seed(0)
  np.random.shuffle(lines)

  for name, data in zip(['train', 'eval'], [lines[: -split], lines[-split: ]]):
    data = sorted(data, key=lambda t: t[0])
    with open('vctk_%s.tsv'%name, 'w') as file:
      writer = csv.writer(file, delimiter='\t')
      writer.writerow(['audio_name', 'transcript'])
      for line in data:
        writer.writerow(line)

  return lines

def create_ng(root, save_name, split):
  lines = []
  print(f"Reading {root} ...")
  with open(root, 'r') as f:
    temp_lines = f.read().splitlines()
    temp_lines = [line.split('\t', 2) for line in temp_lines]
    temp_lines = [('/'.join([root[:root.rfind('/')], a+'.wav']), b) for (a, b) in temp_lines]
    lines += temp_lines
  lines = np.array(lines)
  np.random.seed(0)
  np.random.shuffle(lines)

  for name, data in zip(['train', 'eval'], [lines[: -split], lines[-split: ]]):
    data = sorted(data, key=lambda t: t[0])
    with open('%s_%s.tsv'%(save_name, name), 'w') as file:
      writer = csv.writer(file, delimiter='\t')
      writer.writerow(['audio_name', 'transcript'])
      for line in data:
        writer.writerow(line)

  return lines


if __name__ == '__main__':
  # create_libritts('../LibriTTS/train-clean-100', 'train-clean-100')
  # create_libritts('../LibriTTS/train-clean-360', 'train-clean-360')
  # create_libritts('../LibriTTS/train-other-500', 'train-other-500')
  # create_libritts('../LibriTTS/test-other', 'test-other')
  # create_ng('../en_ng_male/line_index.tsv', 'en_ng_male', split=32)
  # create_ng('../en_ng_female/line_index.tsv', 'en_ng_female', split=32)
  create_vctk('../VCTK-Corpus/wav48', 'vctk', split=128)



