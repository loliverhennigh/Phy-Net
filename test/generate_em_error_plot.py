import math

import numpy as np
import tensorflow as tf
import cv2

import sys
sys.path.append('../')

from model.ring_net import *
from model.loss import *
from model.lattice import *
from utils.experiment_manager import make_checkpoint_path
import random
import time
from tqdm import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn

FLAGS = tf.app.flags.FLAGS

# get restore dir
RESTORE_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)

# shape of test simulation
shape = FLAGS.test_dimensions.split('x')
shape = map(int, shape)

# 2d or not
d2d = False
if len(shape) == 2:
  d2d = True

def calc_mean_and_std(values):
    values_mean = np.sum(values, axis=0) / values.shape[0]
    values_std = np.sqrt(np.sum(np.square(values - np.expand_dims(values_mean, axis=0)), axis=0)/values.shape[0])
    return values_mean, values_std

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    state, boundary = inputs(empty=True, shape=shape)

    # unwrap
    y_1, small_boundary_mul, small_boundary_add, x_2, y_2 = continual_unroll_template(state, boundary)

    # calc mean squared error
    mean_squared_error = tf.nn.l2_loss(state - x_2)

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
    mse_error = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))

    # run simulations
    for sim in tqdm(xrange(FLAGS.test_nr_runs)):
      for run in xrange(FLAGS.test_nr_per_simulation):
        # get frame
        state_feed_dict, boundary_feed_dict = feed_dict(1, shape, FLAGS.lattice_size, sim, run+0)
        fd = {state:state_feed_dict, boundary:boundary_feed_dict}
        y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run([y_1, small_boundary_mul, small_boundary_add], feed_dict=fd)

        # run network 
        for step in tqdm(xrange(FLAGS.test_length)):
          # network step
          state_feed_dict, boundary_feed_dict = feed_dict(1, shape, FLAGS.lattice_size, sim, run+step+0)
          fd = {state:state_feed_dict, boundary:boundary_feed_dict, y_1:y_1_g, small_boundary_mul:small_boundary_mul_g, small_boundary_add:small_boundary_add_g}
          mse, y_1_g = sess.run([mean_squared_error, y_2],feed_dict=fd)
          # calc error
          mse_error[sim, step] = mse

    # step count variable for plotting
    x = np.arange(FLAGS.test_length)

    # mse 
    mse_error_mean, mse_error_std = calc_mean_and_std(mse_error)

    plt.figure(figsize = (6,12))
    plt.style.use('seaborn-darkgrid')

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

    matplotlib.rc('font', **font)

    plt.title(str(shape[0]) + "x" + str(shape[1]) + " EM Simulation", y=1.00, fontsize="x-large")
    plt.errorbar(x, mse_error_mean, yerr=mse_error_std, c='y', capsize=0, lw=0.3)
    plt.xlabel('step', fontsize="x-large")
    plt.ylabel('MSError', fontsize="x-large")
    plt.legend(loc="upper_left")
    plt.savefig("figs/" + str(shape[0]) + "x" + str(shape[1]) + "_2d_em_error_plot.png")

       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
