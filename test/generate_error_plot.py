import math

import numpy as np
import tensorflow as tf
import cv2

import sys
sys.path.append('../')

from model.ring_net import *
from model.loss import *
from utils.experiment_manager import make_checkpoint_path
from utils.numpy_divergence import divergence
from systems.fluid_createTFRecords import generate_feed_dict
import random
import time
from tqdm import *
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

# get restore dir
RESTORE_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)

# shape of test simulation
shape = FLAGS.test_dimensions.split('x')
shape = map(int, shape)

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    state, boundary = inputs(empty=True, shape=shape)

    # unwrap
    y_1, small_boundary_mul, small_boundary_add, x_2, y_2 = continual_unroll_template(state, boundary)

    # restore network
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(RESTORE_DIR)
    if ckpt and ckpt.model_checkpoint_path:
      print("restoring file from " + ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print("no chekcpoint file found from " + RESTORE_DIR + ", this is an error")
      exit()

    # make variables to store error
    mse_error = np.zeros(FLAGS.test_length)
    divergence_true = np.zeros(FLAGS.test_length)
    divergence_generated = np.zeros(FLAGS.test_length)

    # run simulations
    for sim in tqdm(xrange(FLAGS.test_nr_runs)):
      for run in xrange(FLAGS.test_nr_per_simulation):
        # get frame
        state_feed_dict, boundary_feed_dict = generate_feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]), sim, run)
        feed_dict = {state:state_feed_dict, boundary:boundary_feed_dict}
        y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run([y_1, small_boundary_mul, small_boundary_add], feed_dict=feed_dict)

        mse_error_store = np.zeros(FLAGS.test_length)
        divergence_true_store = np.zeros(FLAGS.test_length)
        divergence_generated_store = np.zeros(FLAGS.test_length)
        # run network 
        for step in xrange(FLAGS.test_length):
          # network step
          y_1_g, x_2_g = sess.run([y_2, x_2],feed_dict={y_1:y_1_g, small_boundary_mul:small_boundary_mul_g, small_boundary_add:small_boundary_add_g})
          generated_state = x_2_g[0]
          # get true value
          state_feed_dict, boundary_feed_dict = generate_feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]), sim, run+step)
          true_state = state_feed_dict[0]
          # calc error
          mse_error_store[step] = np.sqrt(np.sum(np.square(generated_state - true_state)))
          # calc divergence
          divergence_true_store[step] = np.sum(np.abs(divergence(true_state[:,:,:-1])))
          divergence_generated_store[step] = np.sum(np.abs(divergence(generated_state[:,:,:-1])))

        # add up calculated values
        mse_error = mse_error + mse_error_store
        divergence_true = divergence_true + divergence_true_store
        divergence_generated = divergence_generated + divergence_generated_store

    plt.figure(1)
    plt.plot(mse_error, label='error')
    plt.title('error')
    plt.legend()
    plt.figure(2)
    plt.plot(divergence_true, label='div true')
    plt.plot(divergence_generated, label='div generated')
    plt.title('divergence')
    plt.legend()
    plt.show()

       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
