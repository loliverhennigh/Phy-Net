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
    percentage_error = np.zeros(FLAGS.test_nr_runs)

    # run simulations
    for sim in tqdm(xrange(FLAGS.test_nr_runs)):
      # get frame
      state_feed_dict, boundary_feed_dict = feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]) + '_test', sim, 100) # hard coded to 100 for start index, This is fine for now
      feed_dict_in = {state:state_feed_dict, boundary:boundary_feed_dict}
      y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run([y_1, small_boundary_mul, small_boundary_add], feed_dict=feed_dict_in)

      true_flow_rate = 0.0
      generated_flow_rate = 0.0
      # run network 
      for step in xrange(FLAGS.test_length):
        # network step
        y_1_g, x_2_g = sess.run([y_2, x_2],feed_dict={y_1:y_1_g, small_boundary_mul:small_boundary_mul_g, small_boundary_add:small_boundary_add_g})
        generated_state = x_2_g[0]
        # get true value
        state_feed_dict, boundary_feed_dict = feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]) + '_test', sim, 100+step)
        true_state = state_feed_dict[0]

        # run throught a line halfway through the simulation and add up the velocitys
        halfway_index = 4*shape[0]/5

        #true_flow_rate += np.sum(true_state[:,:,1]) # y velocity
        #generated_flow_rate += np.sum(generated_state[:,:,1]) # y velocity

        for i in xrange(shape[1]):
          counter = 0
          # add velocity if not boundary
          if boundary_feed_dict[0,i,halfway_index,0] == 0:
            # calc true flow rate
            true_flow_rate += true_state[i,halfway_index,0] # y velocity
            # calc generated flow rate
            generated_flow_rate += generated_state[i,halfway_index,0] # y velocity
            counter += 1
      
        # print things
        #print("generated_flow_rate " + str(generated_flow_rate))
        #print("true_flow_rate " + str(true_flow_rate))

      # print things
      #print("counter " + str(counter))
      #print("true_flow_rate " + str(true_flow_rate))

      # normalize for size of gap and number of steps
      #true_flow_rate = true_flow_rate/(counter*FLAGS.test_length)
      #generated_flow_rate = generated_flow_rate/(counter*FLAGS.test_length)
      true_flow_rate = true_flow_rate/(FLAGS.test_length)
      generated_flow_rate = generated_flow_rate/(FLAGS.test_length)

      # calc percentage error
      percentage_error[sim] = (true_flow_rate - generated_flow_rate)/true_flow_rate
      print("percentage off is " + str(percentage_error[sim]))

    plt.figure(1)
    n, bins, patches = plt.hist(percentage_error, 50, normed=1)
    #l = plt.plot(bins)
    plt.title('error')
    plt.legend()
    plt.show()

       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
