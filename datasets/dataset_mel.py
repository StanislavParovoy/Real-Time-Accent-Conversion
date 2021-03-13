
import logging
import os

import numpy as np
import tensorflow as tf

import sys

# sys.path.append(".")
sys.path.append("..")
# from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio, TFSpeechFeaturizer
from datasets.audio_mel_dataset import AudioMelDataset


class Mel(AudioMelDataset):
  """mel."""

  def __init__(self, 
               n_mels, 
               training=True, 
               **kwargs):
    super().__init__(**kwargs)
    self.n_mels = n_mels
    self.training = training
    self.audio_query = kwargs.get('audio_query').replace('*', '')
    self.mel_query = kwargs.get('mel_query').replace('*', '')

    self.utt_ids = np.array(self.utt_ids)
    self.audio_files = np.array(self.audio_files)
    self.mel_files = np.array(self.mel_files)
    indices = np.arange(len(self.utt_ids))
    np.random.seed(0)
    np.random.shuffle(indices)
    self.utt_ids = self.utt_ids[indices]
    self.audio_files = self.audio_files[indices]
    self.mel_files = self.mel_files[indices]

    cut = 128
    if training:
      self.utt_ids = self.utt_ids[cut: ]
      self.audio_files = self.audio_files[cut: ]
      self.mel_files = self.mel_files[cut: ]
    else:
      self.utt_ids = self.utt_ids[: cut]
      self.audio_files = self.audio_files[: cut]
      self.mel_files = self.mel_files[: cut]

    self.padded_shapes = {
      "utt_ids": [],
      "audios": [None],
      "mels": [None, n_mels],
      "mel_lengths": [],
      "audio_lengths": [],
    }

    # define padded values
    self.padding_values = {
      "utt_ids": "",
      "audios": 0.0,
      "mels": 0.0,
      "mel_lengths": 0,
      "audio_lengths": 0,
    }

  def generator_rand(self, utt_ids):
    indices = np.arange(len(utt_ids))
    np.random.shuffle(indices)
    for i in indices:
      utt_id = utt_ids[i]
      audio_file = self.audio_files[i]
      # mel_file = self.mel_files[i]
      mel_file = audio_file.replace(self.audio_query, self.mel_query)

      items = {
        "utt_ids": utt_id,
        "audio_files": audio_file,
        "mel_files": mel_file,
      }

      yield items

  def create(self,
             batch_size=1,
             map_fn=None,
             **unused_kwargs
  ):
    """Create tf.dataset function."""
    output_types = self.get_output_dtypes()
    datasets = tf.data.Dataset.from_generator(
      self.generator_rand if self.training else self.generator, output_types=output_types, args=(self.get_args())
    )

    # load dataset
    datasets = datasets.map(
      lambda items: self._load_data(items), tf.data.experimental.AUTOTUNE
    )

    datasets = datasets.filter(
      lambda x: x["mel_lengths"] > self.mel_length_threshold
    )
    datasets = datasets.filter(
      lambda x: x["audio_lengths"] > self.audio_length_threshold
    )

    if batch_size > 1 and map_fn is None:
      raise ValueError("map function must define when batch_size > 1.")

    if map_fn is not None:
      datasets = datasets.map(map_fn, tf.data.experimental.AUTOTUNE)

    datasets = datasets.padded_batch(
      batch_size,
      padded_shapes=self.padded_shapes,
      padding_values=self.padding_values,
      drop_remainder=True,
    )
    datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
    return datasets


class MelF0(Mel):
  """mel, log f0."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.f0_query = '_f0.npy'

  @tf.function
  def _load_data(self, items):
    audio = tf.numpy_function(np.load, [items["audio_files"]], tf.float32)
    mel = tf.numpy_function(np.load, [items["mel_files"]], tf.float32)
    f0_name = tf.strings.regex_replace(items['mel_files'], self.mel_query, self.f0_query)
    f0 = tf.numpy_function(np.load, [f0_name], tf.float32)
    mel = tf.squeeze(mel, -1)
    mel = tf.repeat(mel, 4, axis=0)
    mel = tf.concat([mel, f0], axis=-1)

    items = {
      "utt_ids": items["utt_ids"],
      "audios": audio,
      "mels": mel,
      "mel_lengths": len(mel),
      "audio_lengths": len(audio),
    }

    return items


class MelGC(Mel):
  """mel, global condition (e.g. speaker embedding)."""
  def __init__(self, 
               gc_channels=256, 
               **kwargs):
    super().__init__(**kwargs)
    self.gc_query = '_gc.npy'

    self.padded_shapes.update({'gc': [gc_channels]})
    self.padding_values.update({'gc': 0.})

  @tf.function
  def _load_data(self, items):
    audio = tf.numpy_function(np.load, [items["audio_files"]], tf.float32)
    mel = tf.numpy_function(np.load, [items["mel_files"]], tf.float32)
    mel = tf.squeeze(mel, -1)
    gc_name = tf.strings.regex_replace(items['mel_files'], self.mel_query, self.gc_query)
    gc = tf.numpy_function(np.load, [gc_name], tf.float32)

    items = {
      "utt_ids": items["utt_ids"],
      "audios": audio,
      "mels": mel,
      "gc": gc,
      "mel_lengths": len(mel),
      "audio_lengths": len(audio),
    }

    return items

