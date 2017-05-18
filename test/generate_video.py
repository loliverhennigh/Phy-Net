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
from utils.plot_helper import grey_to_short_rainbow
import systems.fluid_createTFRecords as fluid_record
import systems.em_createTFRecords as em_record
#from systems.lattice_utils import *
import random
import time
from tqdm import *
import matplotlib
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

# get restore dir
RESTORE_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)

# video init
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

# shape of test simulation
shape = FLAGS.test_dimensions.split('x')
shape = map(int, shape)

# 2d or not
d2d = False
if len(shape) == 2:
  d2d = True

# open video
if d2d:
  success = video.open('figs/' + str(shape[0]) + "x" + str(shape[1]) + '_2d_video_.mov', fourcc, 16, (3*shape[1], shape[0]), True)
else:
  success = video.open('figs/' + str(shape[0]) + "x" + str(shape[1]) + "x" + str(shape[2]) + '_3d_video.mov', fourcc, 16, (3*shape[2], shape[1]), True)
if success:
  print("opened video stream to fluid_flow.mov")
else:
  print("unable to open video, make sure video settings are correct")
  exit()

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

    state_feed_dict, boundary_feed_dict = feed_dict(1, shape, FLAGS.lattice_size, 0, 0)
    fd = {state:state_feed_dict, boundary:boundary_feed_dict}
    y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run([y_1, small_boundary_mul, small_boundary_add], feed_dict=fd)

    # generate video
    for step in tqdm(xrange(FLAGS.video_length)):
      # calc generated frame compressed state
      state_feed_dict, boundary_feed_dict = feed_dict(1, shape, FLAGS.lattice_size, 0, step)
      fd = {state:state_feed_dict, boundary:boundary_feed_dict, y_1:y_1_g, small_boundary_mul:small_boundary_mul_g, small_boundary_add:small_boundary_add_g}
      v_n_g, v_n_t, y_1_g = sess.run([velocity_norm_generated, velocity_norm_true, y_2],feed_dict=fd)

      # make frame for video
      if not d2d:
        v_n_g = v_n_g[:,10]
        v_n_t = v_n_t[:,10]
      frame = np.concatenate([v_n_g, v_n_t, np.abs(v_n_g - v_n_t)], 2)[0]
      frame = np.uint8(255 * frame/min(.25, np.max(frame)))
      frame = cv2.applyColorMap(frame[:,:,0], 2)

      # write frame to video
      video.write(frame)

    # release video
    video.release()
    cv2.destroyAllWindows()

       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
