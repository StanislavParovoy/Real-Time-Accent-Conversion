import abc
import os
from tqdm import tqdm
from colorama import Fore

import numpy as np
import tensorflow as tf

from tensorflow_asr.configs.config import RunningConfig
from tensorflow_asr.utils.utils import get_num_batches, bytes_to_string, get_reduced_length
from tensorflow_asr.utils.metrics import ErrorRate, wer, cer
from tensorflow_asr.runners.base_runners import BaseRunner, BaseTrainer


class MultiReaderBaseTrainer(BaseTrainer):
    """Customized trainer module for all models."""

    def __init__(self,
                 config: RunningConfig,
                 strategy: tf.distribute.Strategy = None):
        # Configurations
        super().__init__(config)
        self.set_strategy(strategy)
        # Steps and Epochs start from 0
        # Step must be int64 to use tf.summary
        self.steps = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.train_steps_per_epoch = None
        self.eval_steps_per_epoch = None
        # Dataset
        self.train_data_loader = None
        self.eval_data_loader = None
        self.train_reg_data_loader = None
        self.eval_reg_data_loader = None

        with self.strategy.scope():
            self.set_train_metrics()
            self.set_eval_metrics()

    # -------------------------------- GET SET -------------------------------------

    def set_reg_data_loaders(self, 
                             train_reg_dataset, 
                             eval_reg_dataset, 
                             reg_alpha):
        """ Set regularisation data loader (MUST). 
        For simplicity, these dataset generators should be infinite
        """
        self.reg_alpha = reg_alpha

        data = train_reg_dataset.create(self.global_batch_size)
        self.train_reg_data_loader = self.strategy.experimental_distribute_dataset(data)
        self.train_reg_iter = iter(self.train_reg_data_loader)

        data = eval_reg_dataset.create(self.global_batch_size)
        self.eval_reg_data_loader = self.strategy.experimental_distribute_dataset(data)
        self.eval_reg_iter = iter(self.eval_reg_data_loader)

    # -------------------------------- CHECKPOINTS -------------------------------------

    def create_checkpoint_manager(self, max_to_keep=10, **kwargs):
        """Create checkpoint management."""
        with self.strategy.scope():
            self.ckpt = tf.train.Checkpoint(steps=self.steps, **kwargs)
            checkpoint_dir = os.path.join(self.config.outdir, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt, 
                checkpoint_dir, 
                step_counter=self.steps, 
                max_to_keep=max_to_keep)

    def load_checkpoint(self):
        with self.strategy.scope():
            if self.ckpt_manager.latest_checkpoint:
                self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            
    # -------------------------------- RUNNING -------------------------------------

    @tf.function
    def _train_function(self, iterator):
        batch = [(1., next(iterator)), (self.reg_alpha, next(self.train_reg_iter))]
        self.strategy.run(self._train_step, args=(batch, ))

    @tf.function
    def _eval_function(self, iterator):
        batch = [(1., next(iterator)), (self.reg_alpha, next(self.eval_reg_iter))]
        self.strategy.run(self._eval_step, args=(batch, ))

    @abc.abstractmethod
    def _train_step(self, batch):
        """ One step training. Does not return anything"""
        raise NotImplementedError()

    def fit(self, 
            train_dataset, 
            train_reg_dataset, 
            reg_alpha,
            eval_dataset=None, 
            eval_reg_dataset=None, 
            train_bs=None, 
            train_acs=None, 
            eval_bs=None):
        """ Function run start training, including executing "run" func """
        self.set_train_data_loader(train_dataset, train_bs, train_acs)
        self.set_eval_data_loader(eval_dataset, eval_bs)
        self.set_reg_data_loaders(train_reg_dataset, eval_reg_dataset, reg_alpha)
        self.load_checkpoint()
        self.run()

