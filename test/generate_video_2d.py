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
import systems.fluid_createTFRecords as fluid_record
import systems.em_createTFRecords as em_record
#from systems.lattice_utils import *
import random
import time
from tqdm import *

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
  success = video.open('figs/' + str(shape[0]) + "x" + str(shape[1]) + '_2d_video_.mov', fourcc, 4, (3*shape[0], shape[1]), True)
else:
  success = video.open('figs/' + str(shape[0]) + "x" + str(shape[1]) + "x" + str(shape[2]) + '_3d_video.mov', fourcc, 4, (3*shape[2], shape[1]), True)
if success:
  print("opened video stream to fluid_flow.mov")
else:
  print("unable to open video, make sure video settings are correct")
  exit()

def grey_to_short_rainbow(grey):
  a = (1-grey)/0.25
  x = np.floor(a)
  y = np.floor(255*(a-x))
  rainbow = np.zeros((grey.shape[0], grey.shape[1], 3))
  for i in xrange(x.shape[0]):
    for j in xrange(x.shape[1]):
      if x[i,j,0] == 0:
        rainbow[i,j,2] = 255
        rainbow[i,j,1] = y[i,j,0]
        rainbow[i,j,0] = 0
      if x[i,j,0] == 1:
        rainbow[i,j,2] = 255 - y[i,j,0]
        rainbow[i,j,1] = 255
        rainbow[i,j,0] = 0
      if x[i,j,0] == 2:
        rainbow[i,j,2] = 0
        rainbow[i,j,1] = 255
        rainbow[i,j,0] = y[i,j,0] 
      if x[i,j,0] == 3:
        rainbow[i,j,2] = 0
        rainbow[i,j,1] = 255 - y[i,j,0]
        rainbow[i,j,0] = 255
      if x[i,j,0] == 4:
        rainbow[i,j,2] = 0
        rainbow[i,j,1] = 0
        rainbow[i,j,0] = 255
  return rainbow

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    state, boundary = inputs(empty=True, shape=shape)

    # unwrap
    y_1, small_boundary_mul, small_boundary_add, x_2, y_2 = continual_unroll_template(state, boundary)

    # calc velocity
    velocity_generated = lattice_to_vel(x_2)
    velocity_norm_generated = vel_to_norm(velocity_generated)
    velocity_true = lattice_to_vel(state)
    velocity_norm_true = vel_to_norm(velocity_true)

    # calc drag
    drag_generated = lattice_to_force(x_2, boundary)
    drag_norm_generated = drag_generated
    drag_true = lattice_to_force(state, boundary)
    drag_norm_true = drag_true

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
      #v_n_g, v_n_t, d_n_g, d_n_t, y_1_g = sess.run([velocity_norm_generated, velocity_norm_true, drag_norm_generated, drag_norm_true, y_2],feed_dict=fd)
      v_n_g, v_n_t, d_n_g, d_n_t, y_1_g = sess.run([velocity_norm_generated, velocity_norm_true, drag_norm_generated, drag_norm_true, y_2],feed_dict=fd)

      # make frame for video
      frame = np.concatenate([v_n_g, v_n_t, np.abs(v_n_g - v_n_t)], 2)[0]*4.0
      #frame_2 = np.concatenate([d_n_g, d_n_t, np.abs(d_n_g - d_n_t)], 2)*20.0 + .5
      #frame = np.concatenate([frame_2, frame_2], 1)[0]
      frame = np.uint8(grey_to_short_rainbow(frame)) # scale

      # write frame to video
      video.write(frame)

    # release video
    video.release()
    cv2.destroyAllWindows()

       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
