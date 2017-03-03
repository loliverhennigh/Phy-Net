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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FLAGS = tf.app.flags.FLAGS

# get restore dir
RESTORE_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS) 
# shape of test simulation
shape = FLAGS.test_dimensions.split('x')
shape = map(int, shape)

time_sample = [0, 100, 200]

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
    state_feed_dict, boundary_feed_dict = generate_feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1])  + 'x' + str(shape[2]) + '_test', 0, 0)
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

    z_pos = shape[0]/2 -20

    # generate figure
    for step in tqdm(xrange(time_sample[-1]+1)):
      # calc generated frame compressed state
      y_1_g, x_2_g = sess.run([y_2, x_2],feed_dict={y_1:y_1_g, small_boundary_mul:small_boundary_mul_g, small_boundary_add:small_boundary_add_g})

      if step in time_sample:
        # generated frame
        frame_generated = np.sqrt(np.square(x_2_g[0,z_pos,:,:,0]) + np.square(x_2_g[0,z_pos,:,:,1])) #*boundary_max[0,:,:,0:1]
     
        # true frame
        state_feed_dict, boundary_feed_dict = generate_feed_dict(1, shape, FLAGS.lattice_size, 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(shape[2]) + '_test', 0, 0+step)
        flow_true = state_feed_dict[0, z_pos]
        frame_true = np.sqrt(np.square(flow_true[:,:,0]) + np.square(flow_true[:,:,1])) #*boundary_max[0,:,:,0:1]

        # make frame for video
        axarr = plt.subplot(gs1[3*(index)+0])
        axarr.imshow(frame_generated, vmin=0.0, vmax=0.10)
        if index == 0:
          axarr.set_title("Generated", y=0.96)
        axarr.set_ylabel("step " + str(step), y = .5, x = .5)
        axarr.get_xaxis().set_ticks([])
        axarr.get_yaxis().set_ticks([])
        #axarr.axis('off')
        #axarr[index, 0].set_aspect('equal')
        axarr = plt.subplot(gs1[(3*index)+1])
        axarr.imshow(frame_true, vmin=0.0, vmax=0.10)
        if index == 0:
          axarr.set_title("True", y=0.96)
        axarr.get_xaxis().set_ticks([])
        axarr.get_yaxis().set_ticks([])
        #axarr[index, 1].set_aspect('equal')
        axarr = plt.subplot(gs1[(3*index)+2])
        axarr.imshow(np.abs(frame_generated-frame_true), vmin=0.0, vmax=0.10)
        if index == 0:
          axarr.set_title("Difference", y=0.96)
        axarr.get_xaxis().set_ticks([])
        axarr.get_yaxis().set_ticks([])
        #axarr[index, 2].set_aspect('equal')
        index += 1
      
    plt.suptitle(str(shape[0]) + "x" + str(shape[1]) + "x" + str(shape[2])  + " 3D Simulation", fontsize="x-large", y=0.98)
    plt.savefig("figs/" + str(shape[0]) + "x" + str(shape[1]) + "x" + str(shape[2])  + "_3d_flow_image.png")
    plt.show()


       
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
