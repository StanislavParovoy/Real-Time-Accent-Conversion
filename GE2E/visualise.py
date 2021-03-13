from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import io, os
from tensorflow_tts.utils import find_files


def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--save', '-v', default='.')
  parser.add_argument('-n', type=int, default=8)
  parser.add_argument('-m', type=int, default=8)
  parser.add_argument('--dataset', '-d', default='../LibriTTS/train-other-500')
  parser.add_argument('--speaker_info', '-i', default='../LibriTTS/SPEAKERS.TXT')
  args = parser.parse_args()
  return args

def main():
  args = parse_args()

  if 'vctk' in args.dataset.lower():
    depth, start = 2, 1
  elif 'libri' in args.dataset.lower():
    depth, start = 3, 13
  else:
    depth, start = 3, 13
  speaker_files = find_files(args.dataset, '_gc.npy', depth)
  func = lambda filename: filename.split('/')[-depth]
  d = {}
  for f in speaker_files:
    s = func(f)
    if s not in d:
      d[s] = []
    d[s].append(f)
  speaker_files = d
  
  if args.save:
    os.makedirs(args.save, exist_ok=True)
  out_v = io.open('/'.join([args.save, 'vecs.tsv']), 'w', encoding='utf-8')
  out_m = io.open('/'.join([args.save, 'meta.tsv']), 'w', encoding='utf-8')
  with open(args.speaker_info) as file:
    for line in np.random.choice(file.readlines()[start:], size=args.n):
      speaker, info = line.split(maxsplit=1)
      if depth == 2:
        speaker = 'p' + speaker
      if speaker not in speaker_files:
        continue
    
      for i, filename in enumerate(np.random.choice(speaker_files[speaker], 
            size=args.m, 
            replace=args.m < len(speaker_files[speaker]))):
        d_vector = np.load(filename)
        out_m.write(line.replace(speaker, speaker + ' ' + str(i)))
        out_v.write('\t'.join([str(x) for x in d_vector]) + '\n')
  out_v.close()
  out_m.close()

if __name__ == '__main__':
  main()

