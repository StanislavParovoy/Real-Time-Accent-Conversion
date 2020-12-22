import tensorflow as tf
import numpy as np
from tensorflow_asr.datasets.asr_dataset import ASRDataset, ASRSliceDataset


class DatasetNGBase(ASRDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read_entries(self):
        lines = []
        for file_path in self.data_paths:
            print(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                # Skip the header of tsv file
                lines += temp_lines
        # The files is "\t" seperated
        lines = [line.split("\t", 2) for line in lines]
        lines = np.array(lines)
        if self.shuffle:
            np.random.shuffle(lines)  # Mix transcripts.tsv
        self.total_steps = len(lines)
        return lines


class DatasetNG(DatasetNGBase):
    """ Dataset for ASR using Slice """

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
        if len(entries) == 0: return None
        # entries = np.delete(entries, 1, 1)  # Remove unused duration
        dataset = tf.data.Dataset.from_tensor_slices(entries)
        return self.process(dataset, batch_size)

