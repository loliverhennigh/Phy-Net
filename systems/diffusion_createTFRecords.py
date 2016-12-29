
import math

import numpy as np
import scipy.io

import random
import tensorflow as tf 
from glob import glob as glb

import bouncing_balls as b

from tqdm import *


FLAGS = tf.app.flags.FLAGS

# helper function
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_converted_frame(cap, shape):
  ret, frame = cap.read()
  frame = cv2.resize(frame, shape, interpolation = cv2.INTER_CUBIC)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  return frame

def tryint(s):
  try:
    return int(s)
  except:
    return s

def alphanum_key(s):
  return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def generate_tfrecords(seq_length, num_runs, dir_name):

  frame_num = 1
  shape = (32,32)

  if not tf.gfile.Exists(FLAGS.data_dir + 'tfrecords/' + dir_name):
    tf.gfile.MakeDirs(FLAGS.data_dir + 'tfrecords/' + dir_name)

  print('generating records')
  for run in tqdm(xrange(num_runs)):
    filename = FLAGS.data_dir + 'tfrecords/' + dir_name + '/run_' + str(run) + '_seq_length_' + str(seq_length) + '.tfrecords'
  
    tfrecord_filename = glb(FLAGS.data_dir + 'tfrecords/' + dir_name + '/*')  
    if filename not in tfrecord_filename:
   
      writer = tf.python_io.TFRecordWriter(filename)
  
      mat_filenames = glb('../systems/store_' + dir_name + '/run_' + str(run) + '/state_step_*')
      #mat_filenames = glb('../systems/store_' + dir_name + '/run_' + str(run) + '/state_step_*')
      #mat_filenames = nat_filenames.sort(key=alphanum_key)
      num_samples = len(mat_filenames)
      
      ind_dat = 0
      while ind_dat < (num_samples - seq_length):
        seq_frames = np.zeros((seq_length,shape[0],shape[1],frame_num))
        for i in xrange(seq_length):
          diff_state = scipy.io.loadmat('../systems/store_' + dir_name + '/run_' + str(run) + '/state_step_' + str(i+ind_dat+1) + '.mat')
          diff_state = np.array(diff_state['u'])
          diff_state = np.float32(diff_state)
          diff_state = diff_state.reshape([32,32,1])
          seq_frames[i, :, :, :] = diff_state 
        ind_dat = ind_dat + 1
        seq_frames = np.float32(seq_frames)
        seq_frames_flat = seq_frames.reshape([1,seq_length*shape[0]*shape[1]*frame_num])
        seq_frame_raw = seq_frames_flat.tostring()
        # create example and write it
        example = tf.train.Example(features=tf.train.Features(feature={
          'image': _bytes_feature(seq_frame_raw)})) 
        writer.write(example.SerializeToString()) 
    
    
