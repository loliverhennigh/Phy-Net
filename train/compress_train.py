
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

def train():
  """Train ring_net for a number of steps."""

  with tf.Graph().as_default():
    # print important params
    print(FLAGS.system + " system!")
    print("dimensions are " + FLAGS.dimensions + "x" + str(FLAGS.lattice_size))

    # make inputs
    print("Constructing inputs...")
    state, boundary = inputs() 

    # unwrap
    print("Unwrapping network...")
    x_2_o = unroll(state, boundary)

    # apply boundary
    #x_2_o = x_2_o * (1.0-boundary)

    # error
    error_mse = loss_mse(state, x_2_o)
    error_gradient = loss_gradient_difference(state, x_2_o)
    #error_divergence = loss_divergence(x_2_o)
    #error = error_mse + FLAGS.lambda_divergence * error_divergence
    error = error_mse + FLAGS.lambda_divergence * error_gradient

    # train (hopefuly)
    optimizer = set_optimizer(FLAGS.optimizer, FLAGS.reconstruction_lr)
    train_op = optimizer_general(error, optimizer)
    
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
      _ , loss_value = sess.run([train_op, error],feed_dict={})
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
