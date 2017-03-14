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
from systems.lattice_utils import *
import random
import time
from tqdm import *
import matplotlib
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

# get restore dir
RESTORE_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)

# shape of test simulation
shape = FLAGS.test_dimensions.split('x')
shape = map(int, shape)

# lattice properties
lveloc = get_lveloc(FLAGS.lattice_size)

# 2d or not
if len(shape) == 2:
  d2d = True

def calc_mean_and_std(values):
    values_mean = np.sum(values, axis=0) / values.shape[0]
    values_std = np.sqrt(np.sum(np.square(values - np.expand_dims(value_mean, axis=0)), axis=0)/values.shape[0])
    return values_mean, values_std

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
    drag_generated_z = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    drag_true_z = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    flux_generated_x = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    flux_true_x = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    flux_generated_y = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    flux_true_y = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    flux_generated_z = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))
    flux_true_z = np.zeros((FLAGS.test_nr_runs, FLAGS.test_length))


    # run simulations
    for sim in tqdm(xrange(FLAGS.test_nr_runs)):
      for run in xrange(FLAGS.test_nr_per_simulation):
        # get frame
        state_feed_dict, boundary_feed_dict = generate_feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]) + '_test', sim, run+0)
        feed_dict = {state:state_feed_dict, boundary:boundary_feed_dict}
        y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run([y_1, small_boundary_mul, small_boundary_add], feed_dict=feed_dict)

        # run network 
        for step in tqdm(xrange(FLAGS.test_length)):
          # network step
          y_1_g, x_2_g = sess.run([y_2, x_2],feed_dict={y_1:y_1_g, small_boundary_mul:small_boundary_mul_g, small_boundary_add:small_boundary_add_g})
          generated_state = x_2_g[0]
          if d2d:
            generated_state = pad_2d_to_3d(generated_state)
          # get true value
          state_feed_dict, boundary_feed_dict = generate_feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]) + '_test', sim, run+step+0)
          true_state = state_feed_dict[0]
          true_boundary = boundary_feed_dict[0]
          if d2d:
            true_state = pad_2d_to_3d(true_state)
            true_boundary = pad_2d_to_3d(true_boundary)
          # calc error
          mse_error[sim, step] = np.sqrt(np.sum(np.square(generated_state - true_state)))
          # calc divergence
          divergence_true[sim, step] = lattice_to_divergence(true_state,lveloc)
          divergence_generated[sim, step] = lattice_to_divergence(generated_state,lveloc)
          # calc drag
          force_generated = lattice_to_force(generated_state, true_boundary, lveloc)
          drag_generated_x[sim, step] = force_generated[2]
          drag_generated_y[sim, step] = force_generated[1]
          drag_generated_z[sim, step] = force_generated[0]
          force_true = lattice_to_force(true_state, true_boundary, lveloc)
          drag_true_x[sim, step] = force_true[2]
          drag_true_y[sim, step] = force_true[1]
          drag_true_z[sim, step] = force_true[0]
          # calc flux
          flux_generated = np.sum(lattice_to_flux(generated_state, true_boundary, lveloc), axis=(0,1,2))
          flux_generated_x[sim, step] = flux_generated[2]
          flux_generated_y[sim, step] = flux_generated[1]
          flux_generated_z[sim, step] = flux_generated[0]
          flux_true = np.sum(lattice_to_flux(true_state, true_boundary, lveloc), axis=(0,1,2))
          flux_true_x[sim, step] = flux_true[2]
          flux_true_y[sim, step] = flux_true[1]
          flux_true_z[sim, step] = flux_true[0]


    # step count variable for plotting
    x = np.arange(FLAGS.test_length)

    # mse 
    mse_error_mean, mse_error_std = calc_mean_and_std(mse_error)
    # divergence
    divergence_true_mean, divergence_true_std = calc_mean_and_std(divergence_true)
    divergence_generated_mean, divergence_generated_std = calc_mean_and_std(divergence_generated)
    # drag
    drag_true_x_mean, drag_true_x_std = calc_mean_and_std(drag_true_x)
    drag_true_y_mean, drag_true_y_std = calc_mean_and_std(drag_true_y)
    drag_true_z_mean, drag_true_z_std = calc_mean_and_std(drag_true_z)
    drag_generated_x_mean, drag_generated_x_std = calc_mean_and_std(drag_generated_x)
    drag_generated_y_mean, drag_generated_y_std = calc_mean_and_std(drag_generated_y)
    drag_generated_z_mean, drag_generated_z_std = calc_mean_and_std(drag_generated_z)
    # flux
    flux_true_x_mean, flux_true_x_std = calc_mean_and_std(flux_true_x)
    flux_true_y_mean, flux_true_y_std = calc_mean_and_std(flux_true_y)
    flux_true_z_mean, flux_true_z_std = calc_mean_and_std(flux_true_z)
    flux_generated_x_mean, flux_generated_x_std = calc_mean_and_std(flux_generated_x)
    flux_generated_y_mean, flux_generated_y_std = calc_mean_and_std(flux_generated_y)
    flux_generated_z_mean, flux_generated_z_std = calc_mean_and_std(flux_generated_z)


    plt.style.use('seaborn-darkgrid')

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

    matplotlib.rc('font', **font)

    if d2d:
      f, axarr = plt.subplots(6, sharex=True, figsize=(5,7))
      plt.suptitle(str(shape[0]) + "x" + str(shape[1]) + " 2D Simulation", fontsize="x-large", y=0.94)
      axarr[0].errorbar(x, mse_error_mean, yerr=mse_error_std, c='y', capsize=0, lw=0.3)
      axarr[0].set_title('error', y=0.96)
      axarr[1].errorbar(x, divergence_true_mean, yerr=divergence_true_std, label='div true', c='g', capsize=0, lw=0.3)
      axarr[1].errorbar(x, divergence_generated_mean, yerr=divergence_generated_std, label='div generated', c='y', capsize=0, lw=0.2)
      axarr[1].set_title('divergence', y=0.96)
      axarr[2].errorbar(x, drag_true_x_mean, yerr=drag_true_x_std, label='x drag true', c='g', capsize=0, lw=0.3)
      axarr[2].errorbar(x, drag_generated_x_mean, yerr=drag_generated_x_std, label='x drag generated', c='y', capsize=0, lw=0.3)
      axarr[2].set_title('drag x', y=0.96)
      axarr[3].errorbar(x, drag_true_y_mean, yerr=drag_true_y_std, label='y drag true', c='g', capsize=0, lw=0.3)
      axarr[3].errorbar(x, drag_generated_y_mean, yerr=drag_generated_y_std, label='y drag generated', c='y', capsize=0, lw=0.3)
      axarr[3].set_title('drag y', y=0.96)
      axarr[4].errorbar(x, flux_true_x_mean, yerr=flux_true_x_std, label='x flux true', c='g', capsize=0, lw=0.3)
      axarr[4].errorbar(x, flux_generated_x_mean, yerr=flux_generated_x_std, label='x flux generated', c='y', capsize=0, lw=0.3)
      axarr[4].set_title('flux x', y=0.96)
      axarr[5].errorbar(x, flux_true_y_mean, yerr=flux_true_y_std, label='True', c='g', capsize=0, lw=0.3)
      axarr[5].errorbar(x, flux_generated_y_mean, yerr=flux_generated_y_std, label='Generated', c='y', capsize=0, lw=0.3)
      axarr[5].set_title('flux y', y=0.96)
      axarr[5].set_xlabel('step')
      plt.legend(loc="upper_left")
      plt.savefig("figs/" + str(shape[0]) + "x" + str(shape[1]) + "_2d_error_plot.png")
   

    else:
      f, axarr = plt.subplots(8, sharex=True, figsize=(5,8.5))
      plt.suptitle(str(shape[0]) + "x" + str(shape[1]) + "x" + str(shape[2]) + " 3D Simulation", fontsize="x-large", y=0.94)
  
      axarr[0].errorbar(x, mse_error_mean, yerr=mse_error_std, c='y', capsize=0, lw=0.3)
      axarr[0].set_title('error', y=0.96)
  
      axarr[1].errorbar(x, divergence_true_mean, yerr=divergence_true_std, label='div true', c='g', capsize=0, lw=0.3)
      axarr[1].errorbar(x, divergence_generated_mean, yerr=divergence_generated_std, label='div generated', c='y', capsize=0, lw=0.2)
      axarr[1].set_title('divergence', y=0.96)
  
      axarr[2].errorbar(x, drag_true_x_mean, yerr=drag_true_x_std, label='x drag true', c='g', capsize=0, lw=0.3)
      axarr[2].errorbar(x, drag_generated_x_mean, yerr=drag_generated_x_std, label='x drag generated', c='y', capsize=0, lw=0.3)
      axarr[2].set_title('drag x', y=0.96)
  
      axarr[3].errorbar(x, drag_true_y_mean, yerr=drag_true_y_std, label='y drag true', c='g', capsize=0, lw=0.3)
      axarr[3].errorbar(x, drag_generated_y_mean, yerr=drag_generated_y_std, label='y drag generated', c='y', capsize=0, lw=0.3)
      axarr[3].set_title('drag y', y=0.96)
  
      axarr[4].errorbar(x, drag_true_z_mean, yerr=drag_true_z_std, label='z drag true', c='g', capsize=0, lw=0.3)
      axarr[4].errorbar(x, drag_generated_z_mean, yerr=drag_generated_z_std, label='z drag generated', c='y', capsize=0, lw=0.3)
      axarr[4].set_title('drag z', y=0.96)
  
      axarr[5].errorbar(x, flux_true_x_mean, yerr=flux_true_x_std, label='x flux true', c='g', capsize=0, lw=0.3)
      axarr[5].errorbar(x, flux_generated_x_mean, yerr=flux_generated_x_std, label='x flux generated', c='y', capsize=0, lw=0.3)
      axarr[5].set_title('flux x', y=0.96)
  
      axarr[6].errorbar(x, flux_true_y_mean, yerr=flux_true_y_std, label='True', c='g', capsize=0, lw=0.3)
      axarr[6].errorbar(x, flux_generated_y_mean, yerr=flux_generated_y_std, label='Generated', c='y', capsize=0, lw=0.3)
      axarr[6].set_title('flux y', y=0.96)
  
      axarr[7].errorbar(x, flux_true_z_mean, yerr=flux_true_z_std, label='True', c='g', capsize=0, lw=0.3)
      axarr[7].errorbar(x, flux_generated_z_mean, yerr=flux_generated_z_std, label='Generated', c='y', capsize=0, lw=0.3)
      axarr[7].set_title('flux y', y=0.96)
      axarr[7].set_xlabel('step')
  
      plt.legend(loc="upper_left")
      plt.savefig("figs/" + str(shape[0]) + "x" + str(shape[1]) + "x" + str(shape[2]) + "_3d_error_plot.png")
 
    plt.show()

       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
