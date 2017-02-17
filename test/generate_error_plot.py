import math

import numpy as np
import tensorflow as tf
import cv2

import sys
sys.path.append('../')

from model.ring_net import *
from model.loss import *
from utils.experiment_manager import make_checkpoint_path
from utils.numpy_divergence import divergence_2d
from utils.numpy_drag import drag_2d
from utils.numpy_flux import flux_2d
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
    mse_error = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    divergence_true = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    divergence_generated = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    drag_generated_x = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    drag_true_x = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    drag_generated_y = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    drag_true_y = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    flux_generated_x = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    flux_true_x = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    flux_generated_y = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    flux_true_y = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))


    # run simulations
    for sim in tqdm(xrange(FLAGS.test_nr_runs)):
      for run in xrange(FLAGS.test_nr_per_simulation):
        # get frame
        state_feed_dict, boundary_feed_dict = generate_feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]) + '_', sim, run)
        feed_dict = {state:state_feed_dict, boundary:boundary_feed_dict}
        y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run([y_1, small_boundary_mul, small_boundary_add], feed_dict=feed_dict)

        # run network 
        for step in xrange(FLAGS.test_length):
          # network step
          y_1_g, x_2_g = sess.run([y_2, x_2],feed_dict={y_1:y_1_g, small_boundary_mul:small_boundary_mul_g, small_boundary_add:small_boundary_add_g})
          generated_state = x_2_g[0]
          # get true value
          state_feed_dict, boundary_feed_dict = generate_feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]) + '_', sim, run+step)
          true_state = state_feed_dict[0]
          # calc error
          mse_error[sim, step] = np.sqrt(np.sum(np.square(generated_state - true_state)))
          # calc divergence
          divergence_true[sim, step] = np.sum(np.abs(divergence_2d(true_state[:,:,:-1])))
          divergence_generated[sim, step] = np.sum(np.abs(divergence_2d(generated_state[:,:,:-1])))
          # calc drag
          drag_generated_x[sim, step], drag_generated_y[sim, step] = drag_2d(generated_state[:,:,0:2], generated_state[:,:,2], boundary_feed_dict[0,:,:,0])
          drag_true_x[sim, step], drag_true_y[sim, step] = drag_2d(true_state[:,:,0:2], true_state[:,:,2], boundary_feed_dict[0,:,:,0])
          # calc flux
          flux_generated_x[sim, step], flux_generated_y[sim, step] = flux_2d(generated_state[:,:,0:2], generated_state[:,:,2], boundary_feed_dict[0,:,:,0])
          flux_true_x[sim, step], flux_true_y[sim, step] = flux_2d(true_state[:,:,0:2], true_state[:,:,2], boundary_feed_dict[0,:,:,0])


    # step count variable for plotting
    x = np.arange(FLAGS.test_length)

    # mse 
    mse_error_mean = np.sum(mse_error, axis=0)/FLAGS.test_nr_runs
    mse_error_std = np.sqrt(np.sum(np.square(mse_error - np.expand_dims(mse_error_mean, axis=0)), axis=0)/FLAGS.test_nr_runs)
    # divergence
    divergence_true_mean = np.sum(divergence_true, axis=0)/FLAGS.test_nr_runs

    divergence_true_std = np.sqrt(np.sum(np.square(divergence_true - np.expand_dims(divergence_true_mean, axis=0)), axis=0)/FLAGS.test_nr_runs)
    divergence_generated_mean = np.sum(divergence_generated, axis=0)/FLAGS.test_nr_runs

    divergence_generated_std = np.sqrt(np.sum(np.square(divergence_generated - np.expand_dims(divergence_generated_mean, axis=0)), axis=0)/FLAGS.test_nr_runs)
    # drag
    drag_true_x_mean = np.sum(drag_true_x, axis=0)/FLAGS.test_nr_runs

    drag_true_x_std = np.sqrt(np.sum(np.square(drag_true_x - np.expand_dims(drag_true_x_mean, axis=0)), axis=0)/FLAGS.test_nr_runs)
    drag_true_y_mean = np.sum(drag_true_y, axis=0)/FLAGS.test_nr_runs

    drag_true_y_std = np.sqrt(np.sum(np.square(drag_true_y - np.expand_dims(drag_true_y_mean, axis=0)), axis=0)/FLAGS.test_nr_runs)
    drag_generated_x_mean = np.sum(drag_generated_x, axis=0)/FLAGS.test_nr_runs

    drag_generated_x_std = np.sqrt(np.sum(np.square(drag_generated_x - np.expand_dims(drag_generated_x_mean, axis=0)), axis=0)/FLAGS.test_nr_runs)
    drag_generated_y_mean = np.sum(drag_generated_y, axis=0)/FLAGS.test_nr_runs

    drag_generated_y_std = np.sqrt(np.sum(np.square(drag_generated_y - np.expand_dims(drag_generated_y_mean, axis=0)), axis=0)/FLAGS.test_nr_runs)
    # flux
    flux_true_x_mean = np.sum(flux_true_x, axis=0)/FLAGS.test_nr_runs

    flux_true_x_std = np.sqrt(np.sum(np.square(flux_true_x - np.expand_dims(flux_true_x_mean, axis=0)), axis=0)/FLAGS.test_nr_runs)
    flux_true_y_mean = np.sum(flux_true_y, axis=0)/FLAGS.test_nr_runs

    flux_true_y_std = np.sqrt(np.sum(np.square(flux_true_y - np.expand_dims(flux_true_y_mean, axis=0)), axis=0)/FLAGS.test_nr_runs)
    flux_generated_x_mean = np.sum(flux_generated_x, axis=0)/FLAGS.test_nr_runs

    flux_generated_x_std = np.sqrt(np.sum(np.square(flux_generated_x - np.expand_dims(flux_generated_x_mean, axis=0)), axis=0)/FLAGS.test_nr_runs)
    flux_generated_y_mean = np.sum(flux_generated_y, axis=0)/FLAGS.test_nr_runs

    flux_generated_y_std = np.sqrt(np.sum(np.square(flux_generated_y - np.expand_dims(flux_generated_y_mean, axis=0)), axis=0)/FLAGS.test_nr_runs)


    plt.figure(1)
    plt.errorbar(x, mse_error_mean, yerr=mse_error_std)
    #for i in xrange(FLAGS.test_nr_runs):
    #  plt.plot(mse_error[i])
    plt.title('error')
    plt.legend()
    plt.figure(2)
    plt.errorbar(x, divergence_true_mean, yerr=divergence_true_std, label='div true')
    plt.errorbar(x, divergence_generated_mean, yerr=divergence_generated_std, label='div generated')
    plt.title('divergence')
    plt.legend()
    plt.figure(3)
    plt.errorbar(x, drag_true_x_mean, yerr=drag_true_x_std, label='x drag true')
    plt.errorbar(x, drag_generated_x_mean, yerr=drag_generated_x_std, label='x drag generated')
    plt.title('drag x')
    plt.legend()
    plt.figure(4)
    plt.errorbar(x, drag_true_y_mean, yerr=drag_true_y_std, label='y drag true')
    plt.errorbar(x, drag_generated_y_mean, yerr=drag_generated_y_std, label='y drag generated')
    plt.title('drag y')
    plt.legend()
    plt.figure(5)
    plt.errorbar(x, flux_true_x_mean, yerr=flux_true_x_std, label='x flux true')
    plt.errorbar(x, flux_generated_x_mean, yerr=flux_generated_x_std, label='x flux generated')
    #for i in xrange(FLAGS.test_nr_runs):
    #  plt.plot(flux_generated_x[i])
    plt.title('flux x')
    plt.legend()
    plt.figure(6)
    plt.errorbar(x, flux_true_y_mean, yerr=flux_true_y_std, label='y flux true')
    plt.errorbar(x, flux_generated_y_mean, yerr=flux_generated_y_std, label='y flux generated')
    plt.title('flux y')
    plt.legend()

    plt.show()

       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
