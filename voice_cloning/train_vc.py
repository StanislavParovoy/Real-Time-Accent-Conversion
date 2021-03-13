# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train Multi-Band MelGAN."""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
  tf.config.experimental.set_memory_growth(physical_devices[i], True)

import sys
sys.path.append(".")

import argparse
import logging
import os

import numpy as np
import soundfile as sf
import yaml
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import tensorflow_tts
from tensorflow_tts.configs import (
  MultiBandMelGANDiscriminatorConfig,
  MultiBandMelGANGeneratorConfig,
)
from tensorflow_tts.losses import TFMultiResolutionSTFT
from tensorflow_tts.models import (
  TFPQMF,
  TFMelGANMultiScaleDiscriminator,
)
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy

from datasets import MelGC
from rtac.trainers import MultiBandMelganTrainer, collater
from rtac.models import Encoder, MelGANGeneratorVQ
from rtac.losses import VQLoss


class MultiBandMelganVQTrainer(MultiBandMelganTrainer):
  """ Trainer with vector quantisation loss """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.list_metrics_name.append('perplexity')
    self.list_metrics_name.append('vq_loss')

    self.init_train_eval_metrics(self.list_metrics_name)
    self.reset_states_train()
    self.reset_states_eval()

  def compile(self, **kwargs):
    super().compile(**kwargs)
    self.vq_loss = VQLoss(beta=self.config['encoder']['vq_args']['beta'])

  def compute_per_example_generator_losses(self, batch, outputs):
    per_example_losses, dict_metrics_losses = \
      super().compute_per_example_generator_losses(batch, outputs['y_mb_hat'])
    # vq
    vq_loss = self.vq_loss(outputs['z_e'], outputs['e_k'])
    per_example_losses += vq_loss

    dict_metrics_losses.update({"vq_loss": vq_loss})
    dict_metrics_losses.update({"perplexity": outputs['perplexity']})

    return per_example_losses, dict_metrics_losses

  def compute_per_example_discriminator_losses(self, batch, gen_outputs):
    return super().compute_per_example_discriminator_losses(batch, gen_outputs['y_mb_hat'])

  def _one_step_predict_per_replica(self, batch):
    outputs = self._generator(**batch, training=False)['y_mb_hat']
    return outputs


def main():
  """Run training process."""
  parser = argparse.ArgumentParser(
    description="Train MultiBand MelGAN (See detail in examples/multiband_melgan/train_multiband_melgan.py)"
  )
  parser.add_argument(
    "--train-dir", '-td',
    default=None,
    type=str,
    help="directory including training data. ",
  )
  parser.add_argument(
    "--dev-dir", '-dd',
    default=None,
    type=str,
    help="directory including development data. ",
  )
  parser.add_argument(
    "--audio-query", '-aq',
    default='*_wav.npy',
    type=str,
    help="suffix of audio file",
  )
  parser.add_argument(
    "--mel-query", '-mq',
    default='*_mel.npy',
    type=str,
    help="suffix of mel file",
  )
  parser.add_argument(
    "--outdir", '-od',
    type=str, required=True, help="directory to save checkpoints."
  )
  parser.add_argument(
    "--config", '-c',
    type=str, required=True, help="yaml format configuration file."
  )
  parser.add_argument(
    "--resume", '-r',
    default="",
    type=str,
    nargs="?",
    help='checkpoint file path to resume training. (default="")',
  )
  parser.add_argument(
    "--verbose", '-v',
    type=int,
    default=1,
    help="logging level. higher is more logging. (default=1)",
  )
  parser.add_argument(
    "--generator_mixed_precision", '-gmxp',
    default=0,
    type=int,
    help="using mixed precision for generator or not.",
  )
  parser.add_argument(
    "--discriminator_mixed_precision", '-dmxp',
    default=0,
    type=int,
    help="using mixed precision for discriminator or not.",
  )
  parser.add_argument(
    "--pretrained", '-p',
    default="",
    type=str,
    nargs="?",
    help="path of .h5 mb-melgan generator to load weights from",
  )
  args = parser.parse_args()

  # return strategy
  STRATEGY = return_strategy()

  # set mixed precision config
  if args.generator_mixed_precision == 1 or args.discriminator_mixed_precision == 1:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

  args.generator_mixed_precision = bool(args.generator_mixed_precision)
  args.discriminator_mixed_precision = bool(args.discriminator_mixed_precision)

  # set logger
  if args.verbose > 1:
    logging.basicConfig(
      level=logging.DEBUG,
      stream=sys.stdout,
      format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
  elif args.verbose > 0:
    logging.basicConfig(
      level=logging.INFO,
      stream=sys.stdout,
      format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
  else:
    logging.basicConfig(
      level=logging.WARN,
      stream=sys.stdout,
      format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logging.warning("Skip DEBUG/INFO messages")

  # check directory existence
  if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

  # check arguments
  if args.train_dir is None:
    raise ValueError("Please specify --train-dir")
  if args.dev_dir is None:
    raise ValueError("Please specify either --valid-dir")

  # load and save config
  with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)
  with open(config['speech_config']) as f:
    mel_config = yaml.load(f, Loader=yaml.Loader)
  config.update(mel_config)
  config['hop_size'] = config['sample_rate'] * config['stride_ms'] // 1000
  config['sampling_rate'] = config['sample_rate']

  config.update(vars(args))
  config["version"] = tensorflow_tts.__version__
  with open(os.path.join(args.outdir, "config.yml"), "w") as f:
    yaml.dump(config, f, Dumper=yaml.Dumper)
  for key, value in config.items():
    logging.info(f"{key} = {value}")

  # get dataset
  if config["remove_short_samples"]:
    mel_length_threshold = config["batch_max_steps"] // config["hop_size"] \
    + 2 * config["multiband_melgan_generator_params"].get("aux_context_window", 0)
  else:
    mel_length_threshold = None

  audio_query = args.audio_query
  mel_query = args.mel_query
  audio_load_fn = np.load
  mel_load_fn = np.load

  # include global condition
  def collater_gc(items, **kwargs):
    gc = items['gc']
    items = collater(items, **kwargs)
    items['gc'] = gc
    return items

  # define train/valid dataset
  train_dataset = MelGC(
    training=True,
    n_mels=config['n_mels'],
    gc_channels=config['gc_channels'],
    root_dir=args.train_dir,
    audio_query=audio_query,
    mel_query=mel_query,
    audio_load_fn=audio_load_fn,
    mel_load_fn=mel_load_fn,
    mel_length_threshold=mel_length_threshold,
  ).create(
    is_shuffle=config["is_shuffle"],
    map_fn=lambda items: collater_gc(
      items,
      batch_max_steps=tf.constant(config["batch_max_steps"], dtype=tf.int32),
      hop_size=tf.constant(config["hop_size"], dtype=tf.int32),
    ),
    allow_cache=config["allow_cache"],
    batch_size=config["batch_size"]
    * STRATEGY.num_replicas_in_sync
    * config["gradient_accumulation_steps"],
  )

  valid_dataset = MelGC(
    training=False,
    n_mels=config['n_mels'],
    gc_channels=config['gc_channels'],
    root_dir=args.dev_dir,
    audio_query=audio_query,
    mel_query=mel_query,
    audio_load_fn=audio_load_fn,
    mel_load_fn=mel_load_fn,
    mel_length_threshold=mel_length_threshold,
  ).create(
    is_shuffle=config["is_shuffle"],
    map_fn=lambda items: collater_gc(
      items,
      batch_max_steps=tf.constant(
        config["batch_max_steps_valid"], dtype=tf.int32
      ),
      hop_size=tf.constant(config["hop_size"], dtype=tf.int32),
    ),
    allow_cache=config["allow_cache"],
    batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
  )

  # define trainer
  trainer = MultiBandMelganVQTrainer(
    steps=0,
    epochs=0,
    config=config,
    strategy=STRATEGY,
    is_generator_mixed_precision=args.generator_mixed_precision,
    is_discriminator_mixed_precision=args.discriminator_mixed_precision,
  )

  with STRATEGY.scope():
    encoder = Encoder(**config['encoder'])

    generator = MelGANGeneratorVQ(
      encoder=encoder,
      config=MultiBandMelGANGeneratorConfig(
        **config["multiband_melgan_generator_params"]
      ),
      name="multi_band_melgan_generator",
    )
    generator.set_shape(config['n_mels'], config['gc_channels'])

    discriminator = TFMelGANMultiScaleDiscriminator(
      MultiBandMelGANDiscriminatorConfig(
        **config["multiband_melgan_discriminator_params"]
      ),
      name="multi_band_melgan_discriminator",
    )

    pqmf = TFPQMF(
      MultiBandMelGANGeneratorConfig(
        **config["multiband_melgan_generator_params"]
      ),
      dtype=tf.float32,
      name="pqmf",
    )

    # dummy input to build model.
    fake_mels = tf.random.uniform(shape=[1, 100, config['n_mels']], dtype=tf.float32)
    fake_gc = tf.random.uniform(shape=[1, config['gc_channels']], dtype=tf.float32)
    y_mb_hat = generator(mels=fake_mels, gc=fake_gc, training=True)
    for k in y_mb_hat:
      print(k, y_mb_hat[k].shape)
    y_hat = pqmf.synthesis(y_mb_hat['y_mb_hat'])
    print('y_hat:', y_hat.shape)
    discriminator(y_hat)

    if len(args.pretrained) > 1:
      generator.load_weights(args.pretrained)
      logging.info(
        f"Successfully loaded pretrained weight from {args.pretrained}."
      )

    encoder.summary()
    generator.summary()
    discriminator.summary()

    # define optimizer
    generator_lr_fn = getattr(
      tf.keras.optimizers.schedules, config["generator_optimizer_params"]["lr_fn"]
    )(**config["generator_optimizer_params"]["lr_params"])
    discriminator_lr_fn = getattr(
      tf.keras.optimizers.schedules,
      config["discriminator_optimizer_params"]["lr_fn"],
    )(**config["discriminator_optimizer_params"]["lr_params"])

    gen_optimizer = tf.keras.optimizers.Adam(
      learning_rate=generator_lr_fn,
      amsgrad=config["generator_optimizer_params"]["amsgrad"],
    )
    dis_optimizer = tf.keras.optimizers.Adam(
      learning_rate=discriminator_lr_fn,
      amsgrad=config["discriminator_optimizer_params"]["amsgrad"],
    )

  trainer.compile(
    gen_model=generator,
    dis_model=discriminator,
    gen_optimizer=gen_optimizer,
    dis_optimizer=dis_optimizer,
    pqmf=pqmf,
  )

  # start training
  try:
    trainer.fit(
      train_dataset,
      valid_dataset,
      saved_path=os.path.join(config["outdir"], "checkpoints/"),
      resume=args.resume,
    )
  except KeyboardInterrupt:
    trainer.save_checkpoint()
    logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
  main()
