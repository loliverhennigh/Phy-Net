
import math

import numpy as np

import random
import tensorflow as tf 
from glob import glob as glb

import bouncing_balls as b


FLAGS = tf.app.flags.FLAGS

# helper function
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_converted_frame(cap, shape):
  ret, frame = cap.read()
  frame = cv2.resize(frame, shape, interpolation = cv2.INTER_CUBIC)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  return frame

def generate_tfrecords(run_num, num_samples, seq_length, dir_name):

  shape = (32,32)
  frame_num = 3
  T = num_samples 

  filename = FLAGS.data_dir + 'tfrecords/' + dir_name + '/balls_run_num_' + str(run_num) + '_num_samples_' + str(num_samples) + '_seq_length_' + str(seq_length) + '_friction_' + str(FLAGS.friction) + '_num_balls_' + str(FLAGS.num_balls) + '.tfrecords'

  tfrecord_filename = glb(FLAGS.data_dir + 'tfrecords/' + dir_name + '/*')
  if filename in tfrecord_filename:
    print('already a tfrecord there! I will skip this one')
    return

  writer = tf.python_io.TFRecordWriter(filename)

  dat = b.bounce_vec(shape[0], FLAGS.num_balls, T)
  #dat = dat.reshape(T, shape[0], shape[1])
  #dat = np.transpose(dat, (1,2,0))
  #dat = s(dat * 255))

  # normaly it would make sense to put ind_data at 0 however in the case of damped balls its first sequence will always have the same starting high velocity and it will not learn to continue high velocity to low velocity. Basicaly setting the start to be between 0 and seq_length gives better data diversity
  #ind_dat = random.randint(0, seq_length-1)
  ind_dat = 0

  print('now generating tfrecords ' + filename)
 
  while ind_dat < (num_samples - seq_length):
    seq_frames = np.zeros((seq_length,shape[0],shape[1],frame_num))
    for i in xrange(seq_length):
      seq_frames[i, :, :, :] = dat[ind_dat+i,:,:,:]
    ind_dat = ind_dat + 1
    seq_frames = np.float32(seq_frames)
    seq_frames_flat = seq_frames.reshape([1,seq_length*shape[0]*shape[1]*frame_num])
    seq_frame_raw = seq_frames_flat.tostring()
    # create example and write it
    example = tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_feature(seq_frame_raw)})) 
    writer.write(example.SerializeToString()) 



