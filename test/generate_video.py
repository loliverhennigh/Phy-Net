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

RESTORE_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()


shape = FLAGS.test_dimensions.split('x')
shape = map(int, shape)
success = video.open('fluid_flow.mov', fourcc, 4, (3*shape[0], shape[1]), True)

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    state, boundary = inputs(empty=True, shape=shape)

    # unwrap
    y_1, small_boundary_mul, small_boundary_add, x_2, y_2 = continual_unroll(state, boundary)

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
    state_feed_dict, boundary_feed_dict = generate_feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]), 0, 10)
    feed_dict = {state:state_feed_dict, boundary:boundary_feed_dict}
    y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run([y_1, small_boundary_mul, small_boundary_add], feed_dict=feed_dict)

    # Play!!!! 
    for step in tqdm(xrange(FLAGS.video_length)):
      #time.sleep(.5)
      # calc generated frame from t
      y_1_g, x_2_g = sess.run([y_2, x_2],feed_dict={y_1:y_1_g, small_boundary_mul:small_boundary_mul_g, small_boundary_add:small_boundary_add_g})

      frame_generated = np.sqrt(np.square(x_2_g[0,:,:,0:1]) + np.square(x_2_g[0,:,:,1:2])) #*boundary_max[0,:,:,0:1]
      state_feed_dict, boundary_feed_dict = generate_feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]), 0, 10+step)
      flow_true = state_feed_dict[0]

      frame_true = np.sqrt(np.square(flow_true[:,:,0:1]) + np.square(flow_true[:,:,1:2])) #*boundary_max[0,:,:,0:1]
      #frame_generated = np.square(x_2_g[0,:,:,0:1])*10.0 #*boundary_max[0,:,:,0:1]
      #frame_true = np.square(flow_true[0,step+5,:,:,0:1])*10.0 #*boundary_max[0,:,:,0:1]
      #frame_diff = np.abs(frame_true - frame_generated)
      #frame_generated = np.uint8(np.minimum(np.maximum(0, frame_generated*255.0*20.0), 255)) # scale
      #frame_true = np.uint8(np.minimum(np.maximum(0, frame_true*255.0*20.0), 255)) # scale
      #frame_diff = np.uint8(np.minimum(np.maximum(0, frame_diff*255.0*20.0), 255)) # scale
      #print(np.sum(flow_true[0,step+5,:,:,0:1]))
      #print(np.sum(flow_true[0,step+5,:,:,1:2]))
      #print(np.sum(x_2_g[0,:,:,0:1]))
      #print(np.sum(x_2_g[0,:,:,1:2]))
      frame = np.concatenate([frame_generated, frame_true, np.abs(frame_generated - frame_true)], 1)
      frame = np.concatenate([frame, frame, frame], axis=2)
      frame = np.uint8(np.minimum(np.maximum(0, frame*255.0*10.0), 255)) # scale
      video.write(frame)

      #cv2.imshow('frame', frame)
      #cv2.waitKey(0)
      #if cv2.waitKey(1) & 0xFF == ord('q'):
      #  break
    video.release()
    cv2.destroyAllWindows()
    #cv2.destroyAllWindows()

       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
