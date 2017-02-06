
import math
import time

import numpy as np
import scipy.io

import random
import tensorflow as tf 
from glob import glob as glb
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import h5py

from tqdm import *

FLAGS = tf.app.flags.FLAGS

# helper function
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tryint(s):
  try:
    return int(s)
  except:
    return s

def alphanum_key(s):
  return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def load_flow(filename, shape, frame_num):
  # load file
  stream_flow = h5py.File(filename, 'r')

  # process velocity field
  flow_state_vel = np.array(stream_flow['Velocity_0'][:])
  flow_state_vel = flow_state_vel.reshape(shape + [3])
  if len(shape) == 2: # if 2D then kill the z velocity
    flow_state_vel = flow_state_vel[:,:,0:2]

  # process density field
  flow_state_den = np.array(stream_flow['Density_0'][:]) + np.array(stream_flow['Gamma'][:]) - 1.0
  flow_state_den = flow_state_den.reshape(shape + [1])

  # concate state
  flow_state = np.concatenate((flow_state_vel, flow_state_den), len(shape)) 

  # print for testing
  #plt.imshow(flow_state[:,:,0])
  #plt.show()

  return flow_state

def load_boundary(filename, shape, frame_num):
  stream_boundary = h5py.File(filename, 'r')
  boundary_cond = np.array(stream_boundary['Gamma'][:])
  boundary_cond = boundary_cond.reshape([1]+shape+[1])
  return boundary_cond

def generate_feed_dict(seq_length, shape, frame_num, dir_name, run_number, start_index):

  # generate boundry
  boundary_cond = load_boundary(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run_number) + '/fluid_flow_0000.h5', shape, frame_num) # doesnt mater what boundary is loaded

  # generate flow state
  flow_state = np.zeros([seq_length] + shape + [frame_num])
  for i in xrange(seq_length):
    flow_state[i] = load_flow(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run_number) + '/fluid_flow_' + str(start_index + i).zfill(4) + '.h5', shape, frame_num)

  return flow_state, boundary_cond

def generate_tfrecords(seq_length, num_runs, shape, frame_num, dir_name):

  if not tf.gfile.Exists(FLAGS.data_dir + '/tfrecords/' + dir_name):
    tf.gfile.MakeDirs(FLAGS.data_dir + '/tfrecords/' + dir_name)

  for run in tqdm(xrange(num_runs)):
    filename = FLAGS.data_dir + '/tfrecords/' + dir_name + '/run_' + str(run) + '_seq_length_' + str(seq_length) + '.tfrecords'
  
    tfrecord_filename = glb(FLAGS.data_dir + '/tfrecords/' + dir_name + '/*')  

    if filename not in tfrecord_filename:
   
      writer = tf.python_io.TFRecordWriter(filename)
  
    
      h5_filenames = glb(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run) + '/*.h5')
      num_samples = len(h5_filenames)
     
      # first calc boundary (from first sample)
      boundary_cond = load_boundary(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run) + '/fluid_flow_0000.h5', shape, frame_num)
      boundary_cond = np.float32(boundary_cond)
      #boundary_flat = boundary_cond.reshape([1,np.prod(np.array(shape))])
      boundary_flat = boundary_cond.reshape([np.prod(np.array(shape))])
      boundary_raw = boundary_flat.tostring()

      # save tf records
      ind_dat = 0
      while ind_dat < (num_samples - seq_length - 1):
        seq_frames = np.zeros([seq_length] + shape + [frame_num])
        for i in xrange(seq_length):
          t = time.time()
          flow_state = load_flow(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run) + '/fluid_flow_' + str(i+ind_dat).zfill(4) + '.h5', shape, frame_num)
          elapsed = time.time() - t
          #print("time per read is " + str(elapsed))
          
          flow_state = np.float32(flow_state)
          seq_frames[i] = flow_state 
        ind_dat = ind_dat + 1
        seq_frames = np.float32(seq_frames)
        #seq_frames_flat = seq_frames.reshape([1,seq_length*np.prod(np.array(shape))*frame_num])
        seq_frames_flat = seq_frames.reshape([seq_length*np.prod(np.array(shape))*frame_num])
        seq_frame_raw = seq_frames_flat.tostring()
        # create example and write it
        example = tf.train.Example(features=tf.train.Features(feature={
          'flow': _bytes_feature(seq_frame_raw),
          'boundary': _bytes_feature(boundary_raw)}))
        writer.write(example.SerializeToString())
    
    
