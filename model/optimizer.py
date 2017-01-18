
"""functions used to construct different losses
"""

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def set_optimizer(name, lr):
  if name == "adam":
    return tf.train.AdamOptimizer(lr)

def optimizer_discriminator(loss_d, optimizer):
  t_vars = tf.trainable_variables()
  d_vars = [var for var in t_vars if "discriminator" in var.name]
  d_optim = tf.train.AdamOptimizer(FLAGS.gan_lr).minimize(loss_d, var_list=d_vars)
  return d_optim

def optimizer_generator(loss_g, optimizer):
  t_vars = tf.trainable_variables()
  g_vars = [var for var in t_vars if "discriminator" not in var.name]
  g_optim = optimizer.minimize(loss_g, var_list=g_vars)
  return g_optim

def optimizer_general(loss, optimizer):
  optim = optimizer.minimize(loss)
  return optim

