
import math
import time

import numpy as np
import scipy.io

import random
import tensorflow as tf 
from glob import glob as glb
import pylab as pl
from random import randint

from lattice_utils import *

import h5py

from tqdm import *

FLAGS = tf.app.flags.FLAGS

# helper function
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def tryint(s):
  try:
    return int(s)
  except:
    return s

def alphanum_key(s):
  return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def make_feature_from_seq(seq_frames, seq_length, shape, frame_num, name='flow'):
  feature = {}
  for i in xrange(seq_length):
    frame = seq_frames[i]
    frame = np.float32(frame)
    frame = frame.reshape([np.prod(np.array(shape))*frame_num])
    frame = frame.astype(np.float)
    #frame = frame.tostring()
    #frame = frame.tolist()
    feature[name + '/frame_' + str(i)] = _float_feature(frame)
  return feature

def generate_tfrecords(seq_length, num_runs, shape, frame_num, dir_name):

  if not tf.gfile.Exists(FLAGS.tf_data_dir + '/tfrecords/' + dir_name):
    tf.gfile.MakeDirs(FLAGS.tf_data_dir + '/tfrecords/' + dir_name)

  for run in tqdm(xrange(num_runs)):
    filename = FLAGS.tf_data_dir + '/tfrecords/' + dir_name + '/run_' + str(run) + '_seq_length_' + str(seq_length) + '.tfrecords'
  
    tfrecord_filename = glb(FLAGS.tf_data_dir + '/tfrecords/' + dir_name + '/*')  

    if filename not in tfrecord_filename:
   
      writer = tf.python_io.TFRecordWriter(filename)
  
    
      h5_filenames = glb(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run) + '/*.h5')
      num_samples = len(h5_filenames)
     
      # first calc boundary (from first sample)
      boundary_cond = load_boundary(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run) + '/fluid_flow_0000.h5', shape, frame_num)
      boundary_cond = np.float32(boundary_cond)
      #boundary_flat = boundary_cond.reshape([1,np.prod(np.array(shape))])
      boundary_flat = boundary_cond.reshape([np.prod(np.array(shape))])
      #boundary_raw = boundary_flat.tostring()
      boundary_raw = boundary_flat.astype(np.float)

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
        if seq_length > 2:
          ind_dat = ind_dat + (seq_length+1)/2 # this can be made much more efficent but for now this is how it works
        elif seq_length == 2:
          ind_dat += 2
        elif seq_length == 1:
          ind_dat += 1

        # make feature map
        feature = make_feature_from_seq(seq_frames, seq_length, shape, frame_num)
        feature['boundary'] = _float_feature(boundary_raw)

        # create example and write it
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

      writer.close()
    
    
