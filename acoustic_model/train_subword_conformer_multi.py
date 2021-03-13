# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys
sys.path.append('.')
import math, yaml
import argparse
from tensorflow_asr.utils import setup_environment, setup_strategy

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer
from tensorflow_asr.models.conformer import Conformer, Transducer
from tensorflow_asr.optimizers.schedules import TransformerSchedule

from tensorflow_asr.losses.rnnt_losses import rnnt_loss
from tensorflow_asr.utils.utils import get_reduced_length

from datasets import *
from rtac.trainers import MultiReaderBaseTrainer


class MultiReaderTransducerTrainer(MultiReaderBaseTrainer):
    def __init__(self,
                 config,
                 text_featurizer,
                 strategy=None):
        self.text_featurizer = text_featurizer
        super().__init__(config, strategy=strategy)

    def set_train_metrics(self):
        self.train_metrics = {
            "transducer_loss": tf.keras.metrics.Mean("train_transducer_loss", dtype=tf.float32),
            "main_loss": tf.keras.metrics.Mean("train_main_loss", dtype=tf.float32),
            "reg_loss_0": tf.keras.metrics.Mean("train_reg_loss_0", dtype=tf.float32)
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "transducer_loss": tf.keras.metrics.Mean("eval_transducer_loss", dtype=tf.float32),
            "main_loss": tf.keras.metrics.Mean("eval_main_loss", dtype=tf.float32),
            "reg_loss_0": tf.keras.metrics.Mean("eval_reg_loss_0", dtype=tf.float32)
        }
        
    def save_model_weights(self):
        self.model.save_weights(os.path.join(self.config.outdir, "am_%s.h5"%self.steps.numpy()))

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        train_loss = 0.
        names = ['main_loss'] + ['reg_loss_%d' % i for i in range(len(batch))]
        for i, (alpha, r_batch) in enumerate(batch): 
            features, input_length, labels, label_length, prediction, prediction_length = r_batch
            logits = self.model([features, input_length, prediction, prediction_length], training=True)
            per_train_loss = rnnt_loss(
                logits=logits, labels=labels, label_length=label_length,
                logit_length=get_reduced_length(input_length, self.model.time_reduction_factor),
                blank=self.text_featurizer.blank
            )
            self.train_metrics[names[i]].update_state(per_train_loss)
            per_train_loss *= alpha
            train_loss += per_train_loss

        self.train_metrics['transducer_loss'].update_state(train_loss)
        train_loss = tf.nn.compute_average_loss(train_loss,
                                                global_batch_size=self.global_batch_size)

        gradients = tf.gradients(train_loss, self.model.trainable_variables)
        # gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        eval_loss = 0.
        names = ['main_loss'] + ['reg_loss_%d' % i for i in range(len(batch))]
        for i, (alpha, r_batch) in enumerate(batch): 
            features, input_length, labels, label_length, prediction, prediction_length = r_batch

            logits = self.model([features, input_length, prediction, prediction_length], training=False)
            per_eval_loss = rnnt_loss(
                logits=logits, labels=labels, label_length=label_length,
                logit_length=get_reduced_length(input_length, self.model.time_reduction_factor),
                blank=self.text_featurizer.blank
            )
            self.eval_metrics[names[i]].update_state(per_eval_loss)
            per_eval_loss *= alpha
            eval_loss += per_eval_loss

        self.eval_metrics["transducer_loss"].update_state(eval_loss)

    def compile(self,
                model,
                optimizer,
                max_to_keep=10):
        with self.strategy.scope():
            self.model = model
            self.optimizer = tf.keras.optimizers.get(optimizer)
        self.create_checkpoint_manager(max_to_keep, model=self.model, optimizer=self.optimizer)


def main():
    parser = argparse.ArgumentParser(prog="Conformer Training")

    parser.add_argument("--config", type=str, default=DEFAULT_YAML,
                        help="The file path of model configuration file")

    parser.add_argument("--max_ckpts", type=int, default=10,
                        help="Max number of checkpoints to keep")

    parser.add_argument("--tbs", type=int, default=None,
                        help="Train batch size per replica")

    parser.add_argument("--ebs", type=int, default=None,
                        help="Evaluation batch size per replica")

    parser.add_argument("--acs", type=int, default=None,
                        help="Train accumulation steps")

    parser.add_argument("--devices", type=int, nargs="*", default=[0],
                        help="Devices' ids to apply distributed training")

    parser.add_argument("--mxp", default=False, action="store_true",
                        help="Enable mixed precision")

    parser.add_argument("--subwords", type=str, default=None,
                        help="Path to file that stores generated subwords")

    parser.add_argument("--subwords_corpus", nargs="*", type=str, default=[],
                        help="Transcript files for generating subwords")

    parser.add_argument("--train-dir", '-td', nargs='*', default=[
        "en_ng_male_train.tsv",
        "en_ng_female_train.tsv"])
    parser.add_argument("--train-reg-dir", '-trd', nargs='*', default=[
        "libritts_train-clean-100.tsv",
        "libritts_train-clean-360.tsv",
        "libritts_train-other-500.tsv"])
    parser.add_argument("--dev-dir", '-dd', nargs='*', default=[
        "en_ng_male_eval.tsv",
        "en_ng_female_eval.tsv"])
    parser.add_argument("--dev-reg-dir", '-drd', nargs='*', default=[
        "libritts_test-other.tsv"])

    args = parser.parse_args()

    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

    strategy = setup_strategy(args.devices)

    config = Config(args.config, learning=True)
    config.train_dir = args.train_dir
    config.dev_dir = args.dev_dir
    config.train_reg_dir = args.train_reg_dir
    config.dev_reg_dir = args.dev_reg_dir
    with open(config.speech_config) as f:
        speech_config = yaml.load(f, Loader=yaml.Loader)
    speech_featurizer = TFSpeechFeaturizer(speech_config)

    if args.subwords and os.path.exists(args.subwords):
        print("Loading subwords ...")
        text_featurizer = SubwordFeaturizer.load_from_file(config.decoder_config, args.subwords)
    else:
        print("Generating subwords ...")
        text_featurizer = SubwordFeaturizer.build_from_corpus(
            config.decoder_config,
            corpus_files=args.subwords_corpus
        )
        text_featurizer.save_to_file(args.subwords)

    train_dataset = Dataset(
        data_paths=config.train_dir,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        augmentations=config.learning_config.augmentations,
        stage="train", cache=False, shuffle=False
    )
    train_reg_dataset = DatasetInf(
        data_paths=config.train_reg_dir,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        augmentations=config.learning_config.augmentations,
        stage="train", cache=False, shuffle=False
    )
    eval_dataset = Dataset(
        data_paths=config.dev_dir,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        stage="eval", cache=False, shuffle=False
    )
    eval_reg_dataset = DatasetInf(
        data_paths=config.dev_reg_dir,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        augmentations=config.learning_config.augmentations,
        stage="eval", cache=False, shuffle=False
    )

    conformer_trainer = MultiReaderTransducerTrainer(
        config=config.learning_config.running_config,
        text_featurizer=text_featurizer, strategy=strategy
    )

    with conformer_trainer.strategy.scope():
        # build model
        conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
        conformer._build(speech_featurizer.shape)
        conformer.summary(line_length=120)

        optimizer = tf.keras.optimizers.Adam(
            TransformerSchedule(
                d_model=conformer.dmodel,
                warmup_steps=config.learning_config.optimizer_config["warmup_steps"],
                max_lr=(0.05 / math.sqrt(conformer.dmodel))
            ),
            beta_1=config.learning_config.optimizer_config["beta1"],
            beta_2=config.learning_config.optimizer_config["beta2"],
            epsilon=config.learning_config.optimizer_config["epsilon"]
        )

    conformer_trainer.compile(
        model=conformer, 
        optimizer=optimizer,
        max_to_keep=args.max_ckpts)
    conformer_trainer.fit(
        train_dataset, 
        train_reg_dataset, 
        # alpha for regularising dataset; alpha = 1 for training dataset
        1.,
        eval_dataset, 
        eval_reg_dataset, 
        train_bs=args.tbs, eval_bs=args.ebs, train_acs=args.acs)

if __name__ == '__main__':
    main()