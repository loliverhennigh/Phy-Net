
import math

import numpy as np
import scipy.io

import random
import tensorflow as tf 
from glob import glob as glb
import pylab as pl

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
  if frame_num == 3: # if 2D then kill the z velocity
    flow_state_vel = flow_state_vel[:,:,0:2]

  # process density field
  flow_state_den = np.array(stream_flow['Density_0'][:])
  flow_state_den = flow_state_den.reshape(shape + [1])

  # concate state
  flow_state = np.concatenate((flow_state_vel, flow_state_den), len(shape)) 

  return flow_state

def load_boundry(filename, shape, frame_num):
  stream_boundry = h5py.File(filename, 'r')
  boundry_cond = np.array(stream_boundry['Gamma'][:])
  return boundry_cond
  

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
     
      # first calc boundry (from first sample)
      boundry_cond = load_boundry(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run) + '/fluid_flow_0000.h5', shape, frame_num)
      boundry_cond = np.float32(boundry_cond)
      boundry_flat = boundry_cond.reshape([1,np.prod(np.array(shape))])
      boundry_raw = boundry_flat.tostring()

      # save tf records
      ind_dat = 0
      while ind_dat < (num_samples - seq_length - 1):
        seq_frames = np.zeros([seq_length] + shape + [frame_num])
        for i in xrange(seq_length):
          flow_state = load_flow(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run) + '/fluid_flow_' + str(ind_dat).zfill(4) + '.h5', shape, frame_num)
          flow_state = np.float32(flow_state)
          seq_frames[i] = flow_state 
        ind_dat = ind_dat + 1
        seq_frames = np.float32(seq_frames)
        seq_frames_flat = seq_frames.reshape([1,seq_length*np.prod(np.array(shape))*frame_num])
        seq_frame_raw = seq_frames_flat.tostring()
        # create example and write it
        example = tf.train.Example(features=tf.train.Features(feature={
          'flow': _bytes_feature(seq_frame_raw),
          'boundry': _bytes_feature(boundry_raw)})) 
        writer.write(example.SerializeToString()) 
    
    
