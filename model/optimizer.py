
"""functions used to construct different losses
"""

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)

""" switched to optimizer seen in pixel-cnn++
def set_optimizer(name, lr):
  if name == "adam":
    return tf.train.AdamOptimizer(lr)
  elif name == "adagrad":
    return tf.train.AdagradOptimizer(lr)
  elif name == "adadelta":
    return tf.train.AdadeltaOptimizer(lr)

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
"""

