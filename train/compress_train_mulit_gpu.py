
import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
from model.ring_net import *
from model.loss import *
from model.optimizer import *
from utils.experiment_manager import make_checkpoint_path

FLAGS = tf.app.flags.FLAGS

TRAIN_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)

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


nr_gpus = 1

def train():
  """Train ring_net for a number of steps."""

  with tf.Graph().as_default():
    # print important params
    print(FLAGS.system + " system!")
    print("dimensions are " + FLAGS.dimensions + "x" + str(FLAGS.lattice_size))

    # make inputs
    state, boundary = inputs() 
    unroll_template = tf.make_template('unroll_template', unroll)
    x_2_o = unroll_template(state, boundary)

    all_params = tf.trainable_variables()
    ema = tf.train.ExponentialMovingAverage(decay=.9995)
    maintain_averages_op = tf.group(ema.apply(all_params))

    grads = []
    loss_gen = []
    # do for all gpus
    for i in range(nr_gpus):
      # unwrap
      x_2_o = unroll_template(state, boundary)
      error_mse = loss_mse(state, x_2_o)
      error_gradient = loss_gradient_difference(state, x_2_o)
      error = error_mse + FLAGS.lambda_divergence * error_gradient
      loss_gen.append(error)
      # gradients
      grads.append(tf.gradients(loss_gen[i], all_params))

    with tf.device('/gpu:0'):
      for i in range(1, nr_gpus):
        loss_gen[0] += loss_gen[i]
        for j in range(len(grads[0])):
          grads[0][j] += grads[i][j]

      # train (hopefuly)
      optimizer = tf.group(adam_updates(all_params, grads[0], mom1=0.95, mom2=0.9995), maintain_averages_op)

    total_loss = loss_gen[0]
    
    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(variables)   

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # build initialization
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()
    
    # initalize
    sess.run(init)

    # init from seq 1 model
    saver_restore = tf.train.Saver(variables)
    ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
    if ckpt is not None:
      print("init from " + TRAIN_DIR)
      try:
         saver_restore.restore(sess, ckpt.model_checkpoint_path)
      except:
         print("there was a problem using that checkpoint! We will just use random init")

    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph_def=graph_def)

    for step in xrange(FLAGS.max_steps):
      t = time.time()
      _ , loss_value = sess.run([optimizer, total_loss],feed_dict={})
      elapsed = time.time() - t

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step%200 == 0:
        print("loss value at " + str(loss_value))
        print("time per batch is " + str(elapsed))
        summary_str = sess.run(summary_op, feed_dict={})
        summary_writer.add_summary(summary_str, step) 

      if step%2000 == 0:
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + TRAIN_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  if not tf.gfile.Exists(TRAIN_DIR):
    tf.gfile.MakeDirs(TRAIN_DIR)
  if tf.gfile.Exists(TRAIN_DIR) and not FLAGS.restore:
    tf.gfile.DeleteRecursively(TRAIN_DIR)
  train()

if __name__ == '__main__':
  tf.app.run()
