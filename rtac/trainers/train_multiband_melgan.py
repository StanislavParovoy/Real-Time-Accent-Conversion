# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
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
"""Train Multi-Band MelGAN."""

import tensorflow as tf

import sys

sys.path.append(".")

import argparse
import logging
import os

import numpy as np
import soundfile as sf
import yaml

import tensorflow_tts
from rtac.trainers.train_melgan import MelganTrainer
from tensorflow_tts.configs import (
    MultiBandMelGANDiscriminatorConfig,
    MultiBandMelGANGeneratorConfig,
)
from tensorflow_tts.losses import TFMultiResolutionSTFT
from tensorflow_tts.models import (
    TFPQMF,
    TFMelGANGenerator,
    TFMelGANMultiScaleDiscriminator,
)
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy


class MultiBandMelganTrainer(MelganTrainer):
    """Multi-Band MelGAN Trainer class based on MelganTrainer."""

    def __init__(
        self,
        config,
        strategy,
        steps=0,
        epochs=0,
        is_generator_mixed_precision=False,
        is_discriminator_mixed_precision=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_generator_mixed_precision (bool): Use mixed precision for generator or not.
            is_discriminator_mixed_precision (bool): Use mixed precision for discriminator or not.

        """
        super(MultiBandMelganTrainer, self).__init__(
            config=config,
            steps=steps,
            epochs=epochs,
            strategy=strategy,
            is_generator_mixed_precision=is_generator_mixed_precision,
            is_discriminator_mixed_precision=is_discriminator_mixed_precision,
        )

        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "adversarial_loss",
            "subband_spectral_convergence_loss",
            "subband_log_magnitude_loss",
            "fullband_spectral_convergence_loss",
            "fullband_log_magnitude_loss",
            "gen_loss",
            "real_loss",
            "fake_loss",
            "dis_loss",
        ]

        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

    def compile(self, gen_model, dis_model, gen_optimizer, dis_optimizer, pqmf):
        super().compile(gen_model, dis_model, gen_optimizer, dis_optimizer)
        # define loss
        self.sub_band_stft_loss = TFMultiResolutionSTFT(
            **self.config["subband_stft_loss_params"]
        )
        self.full_band_stft_loss = TFMultiResolutionSTFT(
            **self.config["stft_loss_params"]
        )

        # define pqmf module
        self.pqmf = pqmf

    def compute_per_example_generator_losses(self, batch, outputs):
        """Compute per example generator losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        dict_metrics_losses = {}
        per_example_losses = 0.0

        audios = batch["audios"]
        y_mb_hat = outputs
        y_hat = self.pqmf.synthesis(y_mb_hat)

        y_mb = self.pqmf.analysis(tf.expand_dims(audios, -1))
        y_mb = tf.transpose(y_mb, (0, 2, 1))  # [B, subbands, T//subbands]
        y_mb = tf.reshape(y_mb, (-1, tf.shape(y_mb)[-1]))  # [B * subbands, T']

        y_mb_hat = tf.transpose(y_mb_hat, (0, 2, 1))  # [B, subbands, T//subbands]
        y_mb_hat = tf.reshape(
            y_mb_hat, (-1, tf.shape(y_mb_hat)[-1])
        )  # [B * subbands, T']

        # calculate sub/full band spectral_convergence and log mag loss.
        sub_sc_loss, sub_mag_loss = calculate_2d_loss(
            y_mb, y_mb_hat, self.sub_band_stft_loss
        )
        sub_sc_loss = tf.reduce_mean(
            tf.reshape(sub_sc_loss, [-1, self.pqmf.subbands]), -1
        )
        sub_mag_loss = tf.reduce_mean(
            tf.reshape(sub_mag_loss, [-1, self.pqmf.subbands]), -1
        )
        full_sc_loss, full_mag_loss = calculate_2d_loss(
            audios, tf.squeeze(y_hat, -1), self.full_band_stft_loss
        )

        # define generator loss
        gen_loss = 0.5 * (sub_sc_loss + sub_mag_loss) + 0.5 * (
            full_sc_loss + full_mag_loss
        )

        if self.steps >= self.config["discriminator_train_start_steps"]:
            p_hat = self._discriminator(y_hat)
            p = self._discriminator(tf.expand_dims(audios, 2))
            adv_loss = 0.0
            for i in range(len(p_hat)):
                adv_loss += calculate_3d_loss(
                    tf.ones_like(p_hat[i][-1]), p_hat[i][-1], loss_fn=self.mse_loss
                )
            adv_loss /= i + 1
            gen_loss += self.config["lambda_adv"] * adv_loss

            dict_metrics_losses.update({"adversarial_loss": adv_loss},)

        dict_metrics_losses.update({"gen_loss": gen_loss})
        dict_metrics_losses.update({"subband_spectral_convergence_loss": sub_sc_loss})
        dict_metrics_losses.update({"subband_log_magnitude_loss": sub_mag_loss})
        dict_metrics_losses.update({"fullband_spectral_convergence_loss": full_sc_loss})
        dict_metrics_losses.update({"fullband_log_magnitude_loss": full_mag_loss})

        per_example_losses = gen_loss
        return per_example_losses, dict_metrics_losses

    def compute_per_example_discriminator_losses(self, batch, gen_outputs):
        """Compute per example discriminator losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        y_mb_hat = gen_outputs
        y_hat = self.pqmf.synthesis(y_mb_hat)
        (
            per_example_losses,
            dict_metrics_losses,
        ) = super().compute_per_example_discriminator_losses(batch, y_hat)
        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        y_mb_batch_ = self.one_step_predict(batch)  # [B, T // subbands, subbands]
        y_batch = batch["audios"]
        utt_ids = batch["utt_ids"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            y_mb_batch_ = y_mb_batch_.values[0].numpy()
            y_batch = y_batch.values[0].numpy()
            utt_ids = utt_ids.values[0].numpy()
        except Exception:
            y_mb_batch_ = y_mb_batch_.numpy()
            y_batch = y_batch.numpy()
            utt_ids = utt_ids.numpy()

        y_batch_ = self.pqmf.synthesis(y_mb_batch_).numpy()  # [B, T, 1]

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 0):
            # convert to ndarray
            y, y_ = tf.reshape(y, [-1]).numpy(), tf.reshape(y_, [-1]).numpy()

            # plit figure and save it
            utt_id = utt_ids[idx]
            figname = os.path.join(dirname, f"{utt_id}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavefile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(
                figname.replace(".png", "_ref.wav"),
                y,
                self.config["sampling_rate"],
                "PCM_16",
            )
            sf.write(
                figname.replace(".png", "_gen.wav"),
                y_,
                self.config["sampling_rate"],
                "PCM_16",
            )

