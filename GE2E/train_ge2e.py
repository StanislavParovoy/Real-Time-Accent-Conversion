import yaml
import numpy as np
from pathlib import Path
import time, os, sys, shutil, datetime
from argparse import ArgumentParser

sys.path.append(".")
from datasets import Libri
from rtac.models import GE2E
from rtac.losses import SoftmaxLoss
import tensorflow as tf


class Trainer():
  def __init__(self, config):
    super(Trainer, self).__init__()
    np.random.seed(config['training']['seed'])
    tf.random.set_seed(config['training']['seed'])
    self.global_batch_size = config['data']['n'] * config['data']['m']

    self.strategy = get_strategy(config['n_gpu'])

    # data pipeline
    n_mels = config['num_feature_bins']
    self.train_data = Libri(config['train_dataset'], n_mels=n_mels, **config['data'])._build()
    n_frames = (config['data']['min_frames'] + config['data']['max_frames']) / 2
    self.eval_data = Libri(config['eval_dataset'], n_mels=n_mels, n=32, m=10, min_frames=n_frames, max_frames=n_frames)
    self.eval_data = self.eval_data._build()

    if isinstance(self.strategy, tf.distribute.MirroredStrategy):
      self.train_data = self.strategy.experimental_distribute_dataset(self.train_data)
      self.eval_data = self.strategy.experimental_distribute_dataset(self.eval_data)

    self.train_iter, self.eval_iter = iter(self.train_data), iter(self.eval_data)

    # define model, loss, optimisers
    with self.strategy.scope():
      self.model = GE2E(name='ge2e', **config['model'])
      self.model.set_shape(n_mels)
      
      self.loss = SoftmaxLoss(n=config['data']['n'], 
                              m=config['data']['m'],
                              w=self.model.w,
                              b=self.model.b)
      self.eval_loss = SoftmaxLoss(n=32, 
                              m=10,
                              w=self.model.w,
                              b=self.model.b)

      lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
          boundaries=config['training']['boundaries'], values=config['training']['lr'])
      self.opt = tf.keras.optimizers.SGD(lr_schedule)

    # build graph and create variables
    data = next(self.eval_iter)
    with self.strategy.scope():
      a = self.model(data, training=True)
      _ = self.eval_loss(a)
    print('[*** ge2e params: %d ***]' % self.model.count_params())

    self.one = tf.constant(1, dtype=tf.int64)
    self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step')
    self.checkpoint_path = Path(config['save']) / 'model'
    with self.strategy.scope():
      self.checkpoint = tf.train.Checkpoint(
          model=self.model, 
          optimiser=self.opt, 
          global_step=self.global_step)
      self.checkpoint_manager = tf.train.CheckpointManager(
          self.checkpoint,
          self.checkpoint_path,
          max_to_keep=5, 
          step_counter=self.global_step, 
          checkpoint_interval=config['training']['save_interval'])

    self.summary_path = str(Path(config['save']) / 'log')
    self.summary_writer = tf.summary.create_file_writer(self.summary_path)

    shutil.copy(config['config'], Path(config['save']) / 'config.yml')

  def _train_step(self, data):
    self.global_step.assign_add(self.one)

    e = self.model(data, training=True)

    loss, s = self.loss(e)
    accuracy = self.loss.accuracy(s)
    eer = tf.numpy_function(self.loss.eer, [s], tf.float32)

    variables = self.model.trainable_variables
    gradients = tf.gradients(loss, variables)

    grad_var = zip(gradients, variables)
    clipped_grad_var = list(map(clip, grad_var))
    self.opt.apply_gradients(clipped_grad_var)

    losses = {'loss': loss, 
              'accuracy': accuracy, 
              'eer': eer}
    return losses

  @tf.function
  def train_step(self, data):
    loss_per_replica = self.strategy.run(self._train_step, args=[data])
    for k in loss_per_replica:
      loss_per_replica[k] = self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss_per_replica[k], axis=None)
    return loss_per_replica

  def _eval_step(self, data):
    self.global_step.assign_add(self.one)

    e = self.model(data, training=False)

    loss, s = self.eval_loss(e)
    accuracy = self.eval_loss.accuracy(s)
    eer = tf.numpy_function(self.eval_loss.eer, [s], tf.float32)

    losses = {'loss': loss, 
              'accuracy': accuracy, 
              'eer': eer}
    return losses

  @tf.function
  def eval_step(self, data):
    loss_per_replica = self.strategy.run(self._eval_step, args=[data])
    for k in loss_per_replica:
      loss_per_replica[k] = self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss_per_replica[k], axis=None)
    return loss_per_replica

  def log(self, loss_dict, name='train'):
    for k, v in loss_dict.items():
      with self.summary_writer.as_default():
        tf.summary.scalar('/'.join([name, k]), v.numpy(), step=self.global_step)
        self.summary_writer.flush()

  def restore(self):
    with self.strategy.scope():
      self.checkpoint_manager.restore_or_initialize()

  def save(self):
    with self.strategy.scope():
      self.checkpoint_manager.save(checkpoint_number=self.global_step, check_interval=True)
    self.model.save_weights('ge2e_%s.h5'%self.global_step.numpy())


def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--config', '-c', default='config.yml')
  parser.add_argument('--train_dataset', '-td', 
      nargs='*',
      default=['../LibriTTS/train-other-500'])
  parser.add_argument('--eval_dataset', '-ed', 
      nargs='*',
      default=['../LibriTTS/dev-other'])
  parser.add_argument('--n_gpu', '-g', type=int, default=1)
  parser.add_argument('--save', '-v', default='saved_model')
  parser.add_argument('--restore', '-r', default=None)
  args = parser.parse_args()
  return args

def get_strategy(n_gpus):
  devices = tf.config.list_physical_devices("GPU")
  n_gpus = 1
  if n_gpus > 1 and len(devices) > 1:
    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:%d'%i for i in range(n_gpus)])
  elif n_gpus == 1 and len(devices) >= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
  else:
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
  return strategy

def clip(grad_var):
  grad, var = grad_var
  grad_name = grad.name
  grad = tf.clip_by_norm(grad, 3.)
  if 'projection' in var.name:
    grad = 0.5 * grad
  if 'rescale/w' in var.name or 'rescale/b' in var.name:
    grad = 0.01 * grad
  return grad, var

def main():
  args = parse_args()
  with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)
  with open(config['speech_config']) as f:
    mel_config = yaml.load(f, Loader=yaml.Loader)
  config.update(mel_config)
  config.update(vars(args))

  trainer = Trainer(config)
  trainer.restore()
  last_gs = trainer.global_step.numpy()
  print('last global step:', last_gs)

  avg, smooth = 0, 0.3
  for step in range(1, config['training']['steps'] + 1):
    try:
      t = time.time()

      loss_dict = trainer.train_step(next(trainer.train_iter))

      t = time.time() - t
      avg = avg + smooth * (t - avg)
      eta = int((config['training']['steps'] - step) * avg)
      progress = '\r[step %d | %.2f' % (step + last_gs, step / config['training']['steps'] * 100) + '%]'
      info = ' '.join('[%s %.4f]' % (k, loss_dict[k].numpy()) for k in loss_dict)
      timing = '[batch %.3fs] [ETA %s]' % (t, str(datetime.timedelta(seconds=eta)))
      print(' '.join((progress, info, timing)), end='      ')

      if step != 1 and step % config['training']['save_interval'] == 0:
        trainer.save()

      if step != 1 and step % config['training']['log_interval'] == 0:
        trainer.log(loss_dict, 'train')
        loss_dict = trainer.eval_step(next(trainer.eval_iter))
        trainer.log(loss_dict, 'eval')

    except Exception as e:
      if step > 1:
        trainer.save()
      with open('error.txt', 'w') as file:
        file.write(str(e) + '\n')
      raise e
  print()

if __name__ == '__main__':
  main()

