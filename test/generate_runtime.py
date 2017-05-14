import math

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from model.ring_net import *
from model.loss import *
from utils.experiment_manager import make_checkpoint_path
from systems.fluid_createTFRecords import generate_feed_dict
import random
import time
from tqdm import *

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

    # make variable to iterate
    compressed_shape = [x / pow(2,FLAGS.nr_downsamples) for x in shape]
    compressed_state_1 = tf.Variable(np.zeros([1] + compressed_shape + [FLAGS.filter_size_compression], dtype=np.float32), trainable=False) 
    small_boundary_mul_var = tf.Variable(np.zeros([1] + compressed_shape + [FLAGS.filter_size_compression], dtype=np.float32), trainable=False) 
    small_boundary_add_var = tf.Variable(np.zeros([1] + compressed_shape + [FLAGS.filter_size_compression], dtype=np.float32), trainable=False) 

    # make steps to init
    assign_compressed_state_step = tf.group(compressed_state_1.assign(y_1))
    assign_boundary_mul_step = tf.group(small_boundary_mul_var.assign(small_boundary_mul))
    assign_boundary_add_step = tf.group(small_boundary_add_var.assign(small_boundary_add))

    # computation!
    compressed_state_1_boundary = (small_boundary_mul_var * compressed_state_1) + small_boundary_add_var
    compressed_state_2 = compress_template(compressed_state_1_boundary)
    run_step = tf.group(compressed_state_1.assign(compressed_state_2))
    state_out_full = decoding_template(compressed_state_2)
    if len(shape) == 3: 
      state_out_plane = decoding_template(compressed_state_2, extract_type='plane')
    state_out_line = decoding_template(compressed_state_2, extract_type='line')
    state_out_point = decoding_template(compressed_state_2, extract_type='point')
    
    # restore network
    init = tf.global_variables_initializer()
    #variables_to_restore = tf.trainable_variables()
    #saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    sess.run(init)
    #ckpt = tf.train.get_checkpoint_state(RESTORE_DIR)
    #if ckpt and ckpt.model_checkpoint_path:
    #  print("restoring file from " + ckpt.model_checkpoint_path)
    #  saver.restore(sess, ckpt.model_checkpoint_path)
    #else:
    #  print("no chekcpoint file found from " + RESTORE_DIR + ", this is an error")
    #  exit()

    # make fake zero frame to test on
    state_feed_dict = np.zeros([1]+shape+[FLAGS.lattice_size])
    boundary_feed_dict = np.zeros([1]+shape+[1])
    feed_dict = {state:state_feed_dict, boundary:boundary_feed_dict}

    assign_compressed_state_step.run(session=sess, feed_dict=feed_dict)
    assign_boundary_mul_step.run(session=sess, feed_dict=feed_dict)
    assign_boundary_add_step.run(session=sess, feed_dict=feed_dict)
    run_step.run(session=sess)

    # open file to log results (this log file will get directly copied into the paper)
    run_length = 4000
    with open("figs/" + "runtime_log.txt", "a") as myfile:
      # run just compression
      t = time.time()
      for step in tqdm(xrange(run_length)):
        run_step.run(session=sess)
      elapsed = time.time() - t
      time_per_step = elapsed/run_length
      myfile.write(str(shape) + " & %.3f ms " % (time_per_step*1000))
 
      # run with full state out
      t = time.time()
      for step in tqdm(xrange(run_length)):
        state_out_full.eval(session=sess)
      elapsed = time.time() - t
      time_per_step = elapsed/run_length
      myfile.write(" & %.3f ms " % (time_per_step*1000))
      """
   
      # run with plane out
      if len(shape) == 3: 
        t = time.time()
        for step in tqdm(xrange(run_length)):
          state_out_plane.eval(session=sess)
        elapsed = time.time() - t
        time_per_step = elapsed/run_length
        myfile.write(" & %.3f ms " % (time_per_step*1000))
      else:
        myfile.write(" & na")
   
      # run with line out
      t = time.time()
      for step in tqdm(xrange(run_length)):
        state_out_line.eval(session=sess)
      elapsed = time.time() - t
      time_per_step = elapsed/run_length
      myfile.write(" & %.3f ms " % (time_per_step*1000))
   
      # run with point state out
      t = time.time()
      for step in tqdm(xrange(run_length)):
        state_out_point.eval(session=sess)
      elapsed = time.time() - t
      time_per_step = elapsed/run_length
      myfile.write(" & %.3f ms \\\ \n" % (time_per_step*1000))
      """

       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
