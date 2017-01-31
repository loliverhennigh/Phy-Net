
import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
from model.ring_net import *
from model.loss import *
from utils.experiment_manager import make_checkpoint_path

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', '../checkpoints',
                            """dir to store trained net """)

TRAIN_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)

def train():
  """Train ring_net for a number of steps."""

  with tf.Graph().as_default():
    # make inputs
    state, boundry = inputs() 

    # unwrap
    x_2_o = unroll(state, boundry)

    # apply boundry
    x_2_o = x_2_o * (1.0-boundry)
    print(x_2_o.get_shape())
    print(boundry.get_shape())
    print(state.get_shape())

    # error mse
    error_mse = loss_mse(state, x_2_o)
    error_divergence = loss_divergence(x_2_o)
    error = error_mse + 0.0*error_divergence

    # train (hopefuly)
    train_op = tf.train.AdamOptimizer(FLAGS.reconstruction_rl).minimize(error)
    
    # List of all Variables
    variables = tf.all_variables()

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   

    # Summary op
    summary_op = tf.merge_all_summaries()
 
    # build initialization
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()
    
    # initalize
    sess.run(init)

    # init from seq 1 model
    #print("init from " + RESTORE_DIR)
    #saver_restore = tf.train.Saver(variables)
    #ckpt = tf.train.get_checkpoint_state(RESTORE_DIR)
    #saver_restore.restore(sess, ckpt.model_checkpoint_path)

    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(TRAIN_DIR, graph_def=graph_def)

    for step in xrange(20000000):
      t = time.time()
      _ , loss_value = sess.run([train_op, error],feed_dict={})
      elapsed = time.time() - t

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step%100 == 0:
        print("loss value at " + str(loss_value))
        print("time per batch is " + str(elapsed))
        summary_str = sess.run(summary_op, feed_dict={})
        summary_writer.add_summary(summary_str, step) 

      if step%1000 == 0:
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + TRAIN_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(TRAIN_DIR):
    tf.gfile.DeleteRecursively(TRAIN_DIR)
  tf.gfile.MakeDirs(TRAIN_DIR)
  train()

if __name__ == '__main__':
  tf.app.run()
