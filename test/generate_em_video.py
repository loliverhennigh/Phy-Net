
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

FLAGS = tf.app.flags.FLAGS

# get restore dir
RESTORE_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)

# video init
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

# shape of test simulation
shape = FLAGS.test_dimensions.split('x')
shape = map(int, shape)

# make video
success = video.open('figs/' + str(shape[0]) + "x" + str(shape[1]) + '_2d_em_video.mov', fourcc, 8, (3*shape[0], shape[1]), True)

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    state, boundary = inputs(empty=True, shape=shape)

    # unwrap
    y_1, small_boundary_mul, small_boundary_add, x_2, y_2 = continual_unroll_template(state, boundary)

    # calc electric and magnetic fields
    electric_field_generated = lattice_to_electric(x_2, boundary)
    magnetic_field_generated = lattice_to_magnetic(x_2)
    electric_norm_generated = field_to_norm(electric_field_generated)
    magnetic_norm_generated = field_to_norm(magnetic_field_generated)
    electric_field_true = lattice_to_electric(state, boundary)
    magnetic_field_true = lattice_to_magnetic(state)
    electric_norm_true = field_to_norm(electric_field_true)
    magnetic_norm_true = field_to_norm(magnetic_field_true)

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
      x_2_g, y_1_g, m_f_g, m_f_t = sess.run([x_2, y_2, magnetic_norm_generated, magnetic_norm_true],feed_dict=fd)

      m_f_g = m_f_g
      m_f_t = m_f_t
      frame = np.concatenate([m_f_g, m_f_t, np.abs(m_f_g - m_f_t)], 2)[0]
      frame = np.uint8(255 * frame/np.max(frame))
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
