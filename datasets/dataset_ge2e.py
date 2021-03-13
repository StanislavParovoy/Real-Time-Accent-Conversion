from collections import defaultdict
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow_tts.utils import find_files


class Dataset():
  def __init__(self, 
               n_mels=80,    
               n=64,
               m=10,
               min_frames=140,
               max_frames=180):
    self.n_mels = n_mels
    self.n = n
    self.m = m
    self.min_frames = min_frames
    self.max_frames = max_frames

    self.padded_shapes = [None, self.n_mels]
    self.padding_values = 0.

    self._metadata = None

  @property
  def metadata(self):
    raise NotImplementedError

  @property
  def get_speaker_from_filename(self):
    raise NotImplementedError

  def _get_file_of_speaker(self):
    fos = defaultdict(list)
    for file in self.metadata:
      fos[self.get_speaker_from_filename(file)].append(file)
    print('file of speaker:', len(fos), next(iter(fos)))
    # eval
    if self.n is None:
      self.n = len(fos)
    return fos

  def _get_t(self):
    return np.random.randint(low=self.min_frames, high=self.max_frames+1)

  # gen returns batch of n*m filenames, then we use batch size of n*m 
  def generator(self):
    speakers = list(self.file_of_speakers.keys())
    speaker_indices = list(range(len(speakers)))
    while True:
      t = self._get_t()
      for i in np.random.choice(speaker_indices, size=self.n, replace=False):
        speaker_files = self.file_of_speakers[speakers[i]]
        for j in np.random.choice(range(len(speaker_files)), size=self.m, replace=len(speaker_files) < self.m):
          file = speaker_files[j]
          yield {'file': file, 't': t}
          '''
          mel = np.load(file).squeeze()
          if len(mel) < t:
            mel = np.concatenate([mel for _ in range(1 + t // len(mel))], axis=0)
          start = np.random.randint(low=0, high=mel.shape[0]-t+1)
          yield mel[start: start + t]
          '''

  def _load_trim(self, file, t):
    mel = np.load(file).squeeze(-1)
    if len(mel) < t:
      mel = np.concatenate([mel for _ in range(1 + t // len(mel))], axis=0)
    start = np.random.randint(low=0, high=mel.shape[0]-t+1)
    return mel[start: start + t]

  @tf.function
  def _load(self, item):
    mel = tf.numpy_function(self._load_trim, [item['file'], item['t']], tf.float32)
    return mel

  def _build(self):
    self.file_of_speakers = self._get_file_of_speaker()

    dataset = tf.data.Dataset.from_generator(
        self.generator, 
        output_types={'file': tf.string, 't': tf.int32})

    dataset = dataset.map(
        lambda items: self._load(items), tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.batch(self.m * self.n)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class Libri(Dataset):
  def __init__(self, roots, **kwargs):
    super().__init__(**kwargs)
    self._metadata = []
    for root in roots:
      self._metadata += find_files(root, '*_mel.npy')
    self._metadata = sorted(self._metadata)
    self._get_speaker_from_filename = lambda filename: \
      filename.split('/')[-3] if 'libri' in filename.lower() else filename.split('/')[-2]

  @property
  def metadata(self):
    return self._metadata

  @property
  def get_speaker_from_filename(self):
    return self._get_speaker_from_filename

