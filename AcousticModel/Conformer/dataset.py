import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow_asr.datasets.asr_dataset import ASRDataset, ASRSliceDataset
from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio

class DatasetBase(ASRDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio_query = '.wav'
        self.mel_query = '_mel.npy'

    def preprocess(self, audio, transcript):
        with tf.device('/CPU:0'):
            audio = audio.decode("utf-8")
            # features = np.load(audio.replace(self.audio_query, self.mel_query))
            signal = read_raw_audio(audio, self.speech_featurizer.sample_rate)
            signal = self.augmentations.before.augment(signal)
            features = self.speech_featurizer.extract(signal)

            features = self.augmentations.after.augment(features)

            label = self.text_featurizer.extract(transcript.decode("utf-8"))
            label_length = tf.cast(tf.shape(label)[0], tf.int32)
            prediction = self.text_featurizer.prepand_blank(label)
            prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)
            features = tf.convert_to_tensor(features, tf.float32)
            input_length = tf.cast(tf.shape(features)[0], tf.int32)

            return features, input_length, label, label_length, prediction, prediction_length

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
        def generator(entries):
            np.random.shuffle(entries)
            for line in entries:
                yield line
        dataset = tf.data.Dataset.from_generator(generator,
            output_types=tf.string, args=[entries]
        )
        return self.process(dataset, batch_size)


class DatasetLibri(DatasetBase):
    def read_entries(self):
        lines = []
        txt = '.normalized.txt'
        for file_path in self.data_paths:
            print(f"Reading {file_path} ...")
            all_files = map(str, Path(file_path).glob('*/*/*'+txt))
            for filename in all_files:
                with open(filename) as f:
                    text = f.readline()
                audio_name = filename.replace(txt, self.audio_query)
                lines.append((audio_name, text))
        lines = np.array(lines)
        np.random.shuffle(lines)  # Mix transcripts.tsv
        self.total_steps = len(lines)
        return lines


class DatasetVCTK(DatasetBase):
    def read_entries(self):
        lines = []
        txt = '.txt'
        for file_path in self.data_paths:
            print(f"Reading {file_path} ...")
            all_files = map(str, Path(file_path.replace('wav48', 'txt')).glob('*/*'+txt))
            for filename in all_files:
                with open(filename) as f:
                    text = f.readline()
                audio_name = filename.replace('txt', 'wav48').replace(txt, '_wav.npy')
                lines += [(audio_name, text)]
        # lines = [line.split("\t", 2) for line in lines]
        lines = np.array(lines)
        np.random.shuffle(lines)  # Mix transcripts.tsv
        self.total_steps = len(lines)
        return lines


class DatasetNG(DatasetBase):
    def read_entries(self):
        lines = []
        for file_path in self.data_paths:
            print(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                temp_lines = [line.split('\t', 2) for line in temp_lines]
                temp_lines = [('/'.join([file_path, a])+'_wav.npy', b) for (a, b) in temp_lines]
                lines += temp_lines
        # lines = [line.split("\t", 2) for line in lines]
        lines = np.array(lines)
        np.random.shuffle(lines)  # Mix transcripts.tsv
        self.total_steps = len(lines)
        return lines

