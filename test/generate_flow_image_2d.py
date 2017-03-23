import math

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from model.ring_net import *
from model.loss import *
from utils.experiment_manager import make_checkpoint_path
import systems.fluid_createTFRecords as fluid_record
import systems.em_createTFRecords as em_record
from systems.lattice_utils import *
import random
import time
from tqdm import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

time_sample = [0 ,5, 10]

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    state, boundary = inputs(empty=True, shape=shape)
    state = state[0:1,0]
    boundary = boundary[0:1,0]

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
    if FLAGS.system == "fluid":
      frame_name = 'fluid_flow_'
    elif FLAGS.system == "em":
      frame_name = 'em_'
    if d2d:
      frame_name = frame_name + str(shape[0]) + 'x' + str(shape[1]) + '_test'
    else:
      frame_name = frame_name + str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(shape[2]) + '_test'

    lveloc = get_lveloc(FLAGS.lattice_size)
    if FLAGS.system == "fluid":
      state_feed_dict, boundary_feed_dict = fluid_record.generate_feed_dict(1, shape, FLAGS.lattice_size, frame_name, 0, 0)
    elif FLAGS.system == "em":
      state_feed_dict, boundary_feed_dict = em_record.generate_feed_dict(1, shape, FLAGS.lattice_size, frame_name, 0, 0)
    feed_dict = {state:state_feed_dict, boundary:boundary_feed_dict}
    y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run([y_1, small_boundary_mul, small_boundary_add], feed_dict=feed_dict)

    # make plot
    plt.figure(figsize = (len(time_sample), 3))
    gs1 = gridspec.GridSpec(len(time_sample), 3)
    gs1.update(wspace=0.025, hspace=0.025)
    index = 0
    #plt.style.use('seaborn-darkgrid')

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

    matplotlib.rc('font', **font)


    # generate figure
    for step in tqdm(xrange(time_sample[-1]+1)):
      # calc generated frame compressed state
      y_1_g, x_2_g = sess.run([y_2, x_2],feed_dict={y_1:y_1_g, small_boundary_mul:small_boundary_mul_g, small_boundary_add:small_boundary_add_g})

      if step in time_sample:
        # generated frame
        x_2_g = x_2_g[0]
        if FLAGS.system == "fluid":
          if d2d:
            x_2_g = pad_2d_to_3d(x_2_g)
          velocity_generated = lattice_to_vel(x_2_g, lveloc)
          frame_generated = vel_to_norm_vel(velocity_generated)
          if d2d:
            frame_generated = frame_generated[:,:,0,0]
          else:
            frame_generated = frame_generated[0,:,:,0]
        elif FLAGS.system == "em":
          if d2d:
            frame_generated = x_2_g[:,:,0]
          else:
            frame_generated = x_2_g[0,:,:,0]
          frame_generated = np.abs(frame_generated) * 100.0
          print(np.max(frame_generated))
          print("need to implement em stuff")
   
        # true frame
        if FLAGS.system == "fluid":
          state_feed_dict, boundary_feed_dict = fluid_record.generate_feed_dict(1, shape, FLAGS.lattice_size, frame_name, 0, 0+step)
        elif FLAGS.system == "em":
          state_feed_dict, boundary_feed_dict = em_record.generate_feed_dict(1, shape, FLAGS.lattice_size, frame_name, 0, 0+step)
        state_feed_dict = state_feed_dict[0]
        if FLAGS.system == "fluid":
          if d2d:
            state_feed_dict = pad_2d_to_3d(state_feed_dict)
          velocity_true = lattice_to_vel(state_feed_dict, lveloc) #keep first dim on and call it z
          frame_true = vel_to_norm_vel(velocity_true)
          if d2d:
            frame_true = frame_true[:,:,0,0] 
          else:
            frame_true = frame_true[0,:,:,0] 
        elif FLAGS.system == "em":
          if d2d:
            frame_true = state_feed_dict[:,:,0] 
          else:
            frame_true = state_feed_dict[0,:,:,0] 
          frame_true = np.abs(frame_true) * 100.0
          print(np.max(frame_true))
          print("need to implement em stuff")
   


        # make frame for video
        axarr = plt.subplot(gs1[3*(index)+0])
        axarr.imshow(frame_generated, vmin=0.0, vmax=0.18)
        if index == 0:
          axarr.set_title("Generated", y=0.96)
        axarr.set_ylabel("step " + str(step), y = .5, x = .5)
        axarr.get_xaxis().set_ticks([])
        axarr.get_yaxis().set_ticks([])
        #axarr.axis('off')
        #axarr[index, 0].set_aspect('equal')
        axarr = plt.subplot(gs1[(3*index)+1])
        axarr.imshow(frame_true, vmin=0.0, vmax=0.18)
        if index == 0:
          axarr.set_title("True", y=0.96)
        axarr.get_xaxis().set_ticks([])
        axarr.get_yaxis().set_ticks([])
        #axarr[index, 1].set_aspect('equal')
        axarr = plt.subplot(gs1[(3*index)+2])
        axarr.imshow(np.abs(frame_generated-frame_true), vmin=0.0, vmax=0.18)
        if index == 0:
          axarr.set_title("Difference", y=0.96)
        axarr.get_xaxis().set_ticks([])
        axarr.get_yaxis().set_ticks([])
        #axarr[index, 2].set_aspect('equal')
        index += 1
      
    plt.suptitle(str(shape[0]) + "x" + str(shape[1]) + " 2D Simulation", fontsize="x-large", y=0.98)
    plt.savefig("figs/" + str(shape[0]) + "x" + str(shape[1]) + "_2d_flow_image.png")
    print("made it")
    plt.show()


       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
