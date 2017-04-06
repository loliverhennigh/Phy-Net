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

    # add lattice weights to renormalize
    state_add = add_lattice(state)
    x_2_add = add_lattice(x_2)

    # calc mean squared error
    mean_squared_error = tf.nn.l2_loss(state_add - x_2_add)

    # calc divergence
    div_generated = tf.nn.l2_loss(lattice_to_divergence(x_2_add))
    div_true = tf.nn.l2_loss(lattice_to_divergence(state_add))

    # calc flux
    flux_generated = lattice_to_flux(x_2_add, boundary)
    flux_generated = tf.reduce_sum(flux_generated, axis=list(xrange(len(shape)+1)))
    flux_true = lattice_to_flux(state_add, boundary)
    flux_true = tf.reduce_sum(flux_true, axis=list(xrange(len(shape)+1)))

    # calc drag
    drag_generated = lattice_to_force(x_2_add, boundary)
    drag_generated = tf.reduce_sum(drag_generated, axis=list(xrange(len(shape)+1)))
    drag_true = lattice_to_force(state_add, boundary)
    drag_true = tf.reduce_sum(drag_true, axis=list(xrange(len(shape)+1)))

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
        state_feed_dict, boundary_feed_dict = feed_dict(1, shape, FLAGS.lattice_size, sim, run+0)
        fd = {state:state_feed_dict, boundary:boundary_feed_dict}
        y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run([y_1, small_boundary_mul, small_boundary_add], feed_dict=fd)

        # run network 
        for step in tqdm(xrange(FLAGS.test_length)):
          # network step
          state_feed_dict, boundary_feed_dict = feed_dict(1, shape, FLAGS.lattice_size, sim, run+step+0)
          fd = {state:state_feed_dict, boundary:boundary_feed_dict, y_1:y_1_g, small_boundary_mul:small_boundary_mul_g, small_boundary_add:small_boundary_add_g}
          mse, d_g, d_t, f_g, f_t, dr_g, dr_t, y_1_g = sess.run([mean_squared_error, div_generated, div_true, flux_generated, flux_true, drag_generated, drag_true, y_2],feed_dict=fd)
          # get true value
          # calc error
          mse_error[sim, step] = mse
          # calc divergence
          divergence_generated[sim, step] = d_g
          divergence_true[sim, step] = d_t
          # calc drag
          drag_generated_x[sim, step] = dr_g[0]
          drag_generated_y[sim, step] = dr_g[1]
          if not d2d:
            drag_generated_z[sim, step] = dr_g[2]
          drag_true_x[sim, step] = dr_t[0]
          drag_true_y[sim, step] = dr_t[1]
          if not d2d:
            drag_true_z[sim, step] = dr_t[2]
          # calc flux
          flux_generated_x[sim, step] = f_g[0]
          flux_generated_y[sim, step] = f_g[1]
          if not d2d:
            flux_generated_z[sim, step] = f_g[2]
          flux_true_x[sim, step] = f_t[0]
          flux_true_y[sim, step] = f_t[1]
          if not d2d:
            flux_true_z[sim, step] = f_t[2]


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
      axarr[7].set_title('flux z', y=0.96)
      axarr[7].set_xlabel('step')
  
      plt.legend(loc="upper_left")
      plt.savefig("figs/" + str(shape[0]) + "x" + str(shape[1]) + "x" + str(shape[2]) + "_3d_error_plot.png")
 
    plt.show()

       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
