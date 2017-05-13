import math

import numpy as np
import tensorflow as tf
import cv2

import sys
sys.path.append('../')

import os

from model.ring_net import *
from model.loss import *
from model.lattice import *
from utils.experiment_manager import *
import random
import time
from tqdm import *
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool

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

def evaluate_compression_error():

  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    state, boundary = inputs(empty=True, shape=shape)

    # unwrap
    y_1, small_boundary_mul, small_boundary_add, x_2, y_2 = continual_unroll_template(state, boundary)

    # calc mean squared error
    mean_squared_error = tf.nn.l2_loss(state - x_2)
    x_2_add = add_lattice(x_2)
    velocity_generated = lattice_to_vel(x_2_add)

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

    # run simulations
    mse = 0.0
    for sim in tqdm(xrange(FLAGS.test_nr_runs)):
      for step in tqdm(xrange(FLAGS.test_length)):
        # get frame
        state_feed_dict, boundary_feed_dict = feed_dict(1, shape, FLAGS.lattice_size, sim, step+1)
        fd = {state:state_feed_dict, boundary:boundary_feed_dict}
        mse += sess.run(mean_squared_error, feed_dict=fd)
        """
        image = sess.run(velocity_generated, feed_dict=fd)
        plt.figure()
        print(image.shape)
        plt.imshow(image[0,0,:,:,0])
        plt.show()
        """
    mse = mse/(FLAGS.test_nr_runs*FLAGS.test_length)

    # calc compression factor
    compression_size = float(FLAGS.filter_size_compression*np.prod(np.array([x / pow(2,FLAGS.nr_downsamples) for x in shape])))
    flow_size = float(FLAGS.lattice_size*np.prod(np.array(shape)))
    compression_ratio = compression_size/flow_size 

  with open("figs/" + "compression_error_log.txt", "a") as myfile:
    myfile.write(str(len(shape)) + ' ' + str(mse) + ' ' + str(compression_ratio) + "\n")
  

def main(argv=None):  # pylint: disable=unused-argument
  evaluate_compression_error()


if __name__ == '__main__':
  tf.app.run()
