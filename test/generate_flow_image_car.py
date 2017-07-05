
import os
import time

import numpy as np
import tensorflow as tf
import cv2

import sys
sys.path.append('../')

from model.lat_net import *
from model.loss import *
from model.lattice import *
from utils.experiment_manager import *
import systems.fluid_createTFRecords as fluid_record
import systems.em_createTFRecords as em_record

from tqdm import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn

FLAGS = tf.app.flags.FLAGS

# get restore dir
RESTORE_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)

# shape of test simulation
shape = FLAGS.test_dimensions.split('x')
shape = map(int, shape)

# time steps to make images from
time_sample = [100]

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    state, boundary = inputs(empty=True, shape=shape)

    # unwrap
    y_1, small_boundary_mul, small_boundary_add, x_2, y_2 = continual_unroll_template(state, boundary)

    # calc velocity
    x_2_add = add_lattice(x_2)
    state_add = add_lattice(state)
    velocity_generated = lattice_to_vel(x_2_add)
    velocity_norm_generated = vel_to_norm(velocity_generated)
    velocity_true = lattice_to_vel(state_add)
    velocity_norm_true = vel_to_norm(velocity_true)

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

    state_feed_dict, boundary_feed_dict = feed_dict(1, shape, FLAGS.lattice_size, 1, 0)
    fd = {state:state_feed_dict, boundary:boundary_feed_dict}
    y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run([y_1, small_boundary_mul, small_boundary_add], feed_dict=fd)

    # make plot
    plt.figure(figsize = (7, 5.8))
    gs1 = gridspec.GridSpec(3, 2)
    gs1.update(wspace=0.025, hspace=0.025)
    #plt.style.use('seaborn-darkgrid')

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

    matplotlib.rc('font', **font)


    # generate figure
    for car in tqdm(xrange(3)):
      state_feed_dict, boundary_feed_dict = feed_dict(1, shape, FLAGS.lattice_size, car+1, 0)
      fd = {state:state_feed_dict, boundary:boundary_feed_dict}
      y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run([y_1, small_boundary_mul, small_boundary_add], feed_dict=fd)
      for step in tqdm(xrange(time_sample[-1]+1)):
        # calc generated frame compressed state
        state_feed_dict, boundary_feed_dict = feed_dict(1, shape, FLAGS.lattice_size, car+1, step)
        fd = {state:state_feed_dict, boundary:boundary_feed_dict, y_1:y_1_g, small_boundary_mul:small_boundary_mul_g, small_boundary_add:small_boundary_add_g}
        v_n_g, v_n_t, y_1_g = sess.run([velocity_norm_generated, velocity_norm_true, y_2],feed_dict=fd)

        if step in time_sample:
          v_n_g = v_n_g[0,:,:,0]
          v_n_t = v_n_t[0,:,:,0]

          # make images for plot
          axarr = plt.subplot(gs1[2*(car)+0])
          axarr.imshow(v_n_g, vmin=0.0, vmax=0.12)
          if car == 0:
            axarr.set_title("Generated", y=1.00)
          axarr.get_xaxis().set_ticks([])
          axarr.get_yaxis().set_ticks([])
          axarr = plt.subplot(gs1[(2*car)+1])
          axarr.imshow(v_n_t, vmin=0.0, vmax=0.12)
          if car == 0:
            axarr.set_title("True", y=1.00)
          axarr.get_xaxis().set_ticks([])
          axarr.get_yaxis().set_ticks([])
      
    plt.suptitle("Car Simulation", fontsize="x-large", y=0.98)
    plt.savefig("figs/car_2d_flow_image.png")
    print("made it")


       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
