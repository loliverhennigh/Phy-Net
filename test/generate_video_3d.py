import math

import numpy as np
import tensorflow as tf
import cv2

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

# video init
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

# shape of test simulation
shape = FLAGS.test_dimensions.split('x')
shape = map(int, shape)

# open video
success = video.open('figs/' + str(shape[0]) + "x" + str(shape[1]) + "x" + str(shape[2]) + '_3d_video.mov', fourcc, 4, (3*shape[2], shape[1]), True)
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

    # get frame
    state_feed_dict, boundary_feed_dict = generate_feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(shape[2]) + '_test', 0, 0)
    feed_dict = {state:state_feed_dict, boundary:boundary_feed_dict}
    y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run([y_1, small_boundary_mul, small_boundary_add], feed_dict=feed_dict)
    last_step_frame_true = 0.0

    # generate video
    for step in tqdm(xrange(FLAGS.video_length)):
      # calc generated frame compressed state
      y_1_g, x_2_g = sess.run([y_2, x_2],feed_dict={y_1:y_1_g, small_boundary_mul:small_boundary_mul_g, small_boundary_add:small_boundary_add_g})

      # normalize velocity
      #frame_generated = np.sqrt(np.square(x_2_g[0,:,:,2:3])) #*boundary_max[0,:,:,0:1]
      #frame_generated = np.sqrt(np.square(x_2_g[0,:,:,0:1])) #*boundary_max[0,:,:,0:1]
      index = shape[0]/2 + 10
      frame_generated = np.sqrt(np.square(x_2_g[0,index,:,:,0:1]) + np.square(x_2_g[0,index,:,:,1:2]) + np.square(x_2_g[0,index,:,:,2:3])) #*boundary_max[0,:,:,0:1]
     
      # get true normalized velocity 
      state_feed_dict, boundary_feed_dict = generate_feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(shape[2]) + '_test', 0, 0+step)
      flow_true = state_feed_dict[0]
      frame_true = np.sqrt(np.square(flow_true[index,:,:,0:1]) + np.square(flow_true[index,:,:,1:2]) + np.square(flow_true[index,:,:,2:3])) #*boundary_max[0,:,:,0:1]
      print(np.sum(np.abs(frame_true - last_step_frame_true)))
      last_step_frame_true = frame_true
      #frame_generated = np.sqrt(np.square(flow_true[index,:,:,0:1]) + np.square(flow_true[index,:,:,1:2]) + np.square(flow_true[index,:,:,2:3])) #*boundary_max[0,:,:,0:1]
      #frame_true = np.sqrt(np.square(flow_true[:,:,2:3])) #*boundary_max[0,:,:,0:1]
      #frame_true = np.sqrt(np.square(flow_true[:,:,0:1])) #*boundary_max[0,:,:,0:1]

      # make frame for video
      frame = np.concatenate([frame_generated, frame_true, np.abs(frame_generated - frame_true)], 1)
      frame = np.concatenate([frame, frame, frame], axis=2)
      frame = np.uint8(np.minimum(np.maximum(0, frame*255.0*10.0), 255)) # scale

      # write frame to video
      video.write(frame)

    # release video
    video.release()
    cv2.destroyAllWindows()

       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
