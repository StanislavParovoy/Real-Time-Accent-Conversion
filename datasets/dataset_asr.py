import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow_asr.datasets.asr_dataset import ASRDataset, ASRSliceDataset, ASRSliceTestDataset
from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio


class Dataset(ASRDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, path, transcript):
      return super().preprocess(path.decode("utf-8"), transcript)

    @tf.function
    def parse(self, record):
      return tf.numpy_function(
        self.preprocess,
        inp=[record[0], record[1]],
        Tout=[tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32]
      )

    def create(self, batch_size):
      entries = self.read_entries()
      print('entries:', len(entries), entries[0])
      if len(entries) == 0: return None
      def generator():
        np.random.shuffle(entries)
        for line in entries:
            yield line
      dataset = tf.data.Dataset.from_generator(generator,
          output_types=tf.string, args=[]
      )
      return self.process(dataset, batch_size)


class DatasetInf(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create(self, batch_size):
      entries = self.read_entries()
      print('entries:', len(entries), entries[0])
      if len(entries) == 0: return None
      def generator():
        while True:
          np.random.shuffle(entries)
          for line in entries:
              yield line
      dataset = tf.data.Dataset.from_generator(generator,
          output_types=tf.string, args=[]
      )
      return self.process(dataset, batch_size)


class ASRSliceTestDataset(ASRDataset):
    def preprocess(self, path, transcript):
        with tf.device("/CPU:0"):
            signal = read_raw_audio(path.decode("utf-8"), self.speech_featurizer.sample_rate)

            features = self.speech_featurizer.extract(signal)
            features = tf.convert_to_tensor(features, tf.float32)
            input_length = tf.cast(tf.shape(features)[0], tf.int32)

            label = self.text_featurizer.extract(transcript.decode("utf-8"))
            label = tf.convert_to_tensor(label, dtype=tf.int32)

            return path, features, input_length, label

    @tf.function
    def parse(self, record):
        return tf.numpy_function(
            self.preprocess,
            inp=[record[0], record[1]],
            Tout=[tf.string, tf.float32, tf.int32, tf.int32]
        )

    def process(self, dataset, batch_size):
        dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(TFRECORD_SHARDS, reshuffle_each_iteration=True)

        # PADDED BATCH the dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                tf.TensorShape([]),
                tf.TensorShape(self.speech_featurizer.shape),
                tf.TensorShape([]),
                tf.TensorShape([None]),
            ),
            padding_values=("", 0.0, 0, self.text_featurizer.blank),
            drop_remainder=True
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        self.total_steps = self.total_steps // batch_size
        return dataset

    def create(self, batch_size):
        entries = self.read_entries()
        if len(entries) == 0: return None
        dataset = tf.data.Dataset.from_tensor_slices(entries)
        return self.process(dataset, batch_size)

