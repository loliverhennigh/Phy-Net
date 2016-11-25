
import math

import numpy as np
import scipy.io

import random
import tensorflow as tf 
from glob import glob as glb
import pylab as pl

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

def load_flow(filename, shape, frame_num):
  # load and reshape
  fluid_state = np.loadtxt(filename)
  fluid_state = fluid_state.reshape(shape[0], shape[1], frame_num)
  return fluid_state

def load_boundry(filename, shape, frame_num):
  # load and reshape
  ball_location= np.loadtxt(filename)
  b1x = ball_location[0]
  b1y = ball_location[1]
  b2x = ball_location[2]
  b2y = ball_location[3]
  b3x = ball_location[4]
  b3y = ball_location[5]
  b4x = ball_location[6]
  b4y = ball_location[7]

  # calc signed distance function.
  boundry = np.zeros((shape[0], shape[1], 2))
  for x in xrange(shape[0]):
    for y in xrange(shape[1]):
      # distance ball 1
      d1 = np.sqrt(np.square(b1x - x) + np.square(b1y - y))
      d2 = np.sqrt(np.square(b2x - x) + np.square(b2y - y))
      d3 = np.sqrt(np.square(b3x - x) + np.square(b3y - y))
      d4 = np.sqrt(np.square(b4x - x) + np.square(b4y - y))
      dw1 = y
      dw2 = shape[1] - y
      min_d = np.min([np.min([d1, d2, d3, d4]) - 10, dw1, dw2])
      #if min_d < 0: # 10 because the balls are size 10
      #  min_d = -min_d 
      boundry[x,y,0] = min_d
      boundry[x,y,1] = x
  #print(boundry[b1x,b1y])
  #print(boundry[b1x-10,b1y])
  #pl.imshow(boundry[:,:,0])
  #pl.show()
  #pl.imshow(boundry[:,:,1])
  #pl.show()
      
  return boundry
  

def generate_tfrecords(seq_length, num_runs, shape, frame_num, dir_name):

  if not tf.gfile.Exists(FLAGS.data_dir + 'tfrecords/' + dir_name):
    tf.gfile.MakeDirs(FLAGS.data_dir + 'tfrecords/' + dir_name)

  print('generating records')
  for run in tqdm(xrange(num_runs)):
    filename = FLAGS.data_dir + 'tfrecords/' + dir_name + '/run_' + str(run) + '_seq_length_' + str(seq_length) + '.tfrecords'
  
    tfrecord_filename = glb(FLAGS.data_dir + 'tfrecords/' + dir_name + '/*')  

    if filename not in tfrecord_filename:
   
      writer = tf.python_io.TFRecordWriter(filename)
  
    
      mat_filenames = glb('../systems/store_' + dir_name + '/sam' + str(run) + '/run*')
      num_samples = len(mat_filenames)
      
      # first calc boundry
      boundry_cond = load_boundry('../systems/store_' + dir_name + '/sam' + str(run) + '/run', shape, frame_num)
      boundry_cond = np.float32(boundry_cond)
      boundry_flat = boundry_cond.reshape([1,shape[0]*shape[1]*2])
      boundry_raw = boundry_flat.tostring()

      # save tf records
      ind_dat = 0
      while ind_dat < (num_samples - seq_length - 1):
        print("read!")
        seq_frames = np.zeros((seq_length,shape[0],shape[1],frame_num))
        for i in xrange(seq_length):
          flow_state = load_flow('../systems/store_' + dir_name + '/sam' + str(run) + '/run' + str(i+ind_dat+1) + '.data', shape, frame_num)
          flow_state = np.float32(flow_state)
          seq_frames[i, :, :, :] = flow_state 
        ind_dat = ind_dat + 1
        seq_frames = np.float32(seq_frames)
        seq_frames_flat = seq_frames.reshape([1,seq_length*shape[0]*shape[1]*frame_num])
        seq_frame_raw = seq_frames_flat.tostring()
        # create example and write it
        example = tf.train.Example(features=tf.train.Features(feature={
          'flow': _bytes_feature(seq_frame_raw),
          'boundry': _bytes_feature(boundry_raw)})) 
        writer.write(example.SerializeToString()) 
    
    
