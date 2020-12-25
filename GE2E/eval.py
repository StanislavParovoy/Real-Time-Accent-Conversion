from model import GE2E
from dataset import *
from utils import *
import parameters
from train import parse_args
import tensorflow as tf
import time, os, sys, json
from argparse import ArgumentParser

def eval(sess, model, loss):
  s, a, l = sess.run([model.s, model.accuracy, loss])
  eer = model._eer(s)
  print('[eer %.3f] [accuracy %.3f] [loss %.4f]' % (eer, a, l))
  # print(s)

def main():
  # this will update N and M, if provided from command line
  args = parse_args()
  if 'vctk' in args.dataset.lower():
    dataset = VCTK(data_path=args.dataset)
  else:
    dataset = LibriSpeech(data_path=args.dataset)

  model = GE2E(scope='ge2e', training=False)
  _ = model(dataset.x)
  loss, _ = model.compute_loss()
  learning_rate = tf.compat.v1.Variable(tf.constant(0, dtype=tf.float32), trainable=False, name='learning_rate')
  global_step = tf.compat.v1.Variable(tf.constant(0, dtype=tf.int32), trainable=False, name='global_step')
  for key, value in parameters.learning_rate_schedule.items():
    learning_rate = tf.compat.v1.cond(tf.compat.v1.less(global_step, int(key)), 
      lambda: learning_rate, lambda: tf.compat.v1.constant(value))

  # restore pretrained model
  sess = tf.compat.v1.Session()
  vars = {}
  for v in tf.compat.v1.global_variables():
    vars[v.name.replace('lstm_cell/', '').replace(':0', '')] = v
  saver = tf.compat.v1.train.Saver(vars)
  latest_checkpoint = tf.train.latest_checkpoint(args.restore)
  if latest_checkpoint is not None:
    saver.restore(sess, latest_checkpoint)
  else:
    sess.run(tf.compat.v1.global_variables_initializer())
  sess.run(dataset.init)

  lr, gs = sess.run([learning_rate, global_step])
  print('[restore] last global step: %d, learning rate: %.5f' % (gs, lr))
  
  for i in range(3):
    print('test', i)
    eval(sess, model, loss)

if __name__ == '__main__':
  suppress_tf_warning()
  main()

