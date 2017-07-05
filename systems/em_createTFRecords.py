
import numpy as np
import tensorflow as tf 
from glob import glob as glb
import h5py
from tqdm import *

from model.lattice import *

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

def load_em(filename, shape, frame_num):
  # load file
  stream_em = h5py.File(filename, 'r')

  # process lattice state
  em_state = np.array(stream_em['State'][:])
  if len(shape) == 2:
    shape = [1] + shape
  em_state = 10.0*em_state.reshape(shape + [frame_num])
  stream_em.close()

  return em_state

def load_boundary(filename, shape, frame_num):
  stream_boundary = h5py.File(filename, 'r')
  boundary_cond = np.array(stream_boundary['Epsilon'][:])
  boundary_cond = boundary_cond.reshape([1]+shape+[1])
  stream_boundary.close()
  return boundary_cond

def make_feature_from_seq(seq_frames, seq_length, shape, frame_num):
  feature = {}
  for i in xrange(seq_length):
    frame = seq_frames[i]
    frame = np.float32(frame)
    frame = frame.reshape([np.prod(np.array(shape))*frame_num])
    frame = frame.astype(np.float)
    feature['em/frame_' + str(i)] = _float_feature(frame)
  return feature

def generate_feed_dict(seq_length, shape, frame_num, dir_name, run_number, start_index):

  # generate boundry
  boundary_cond = load_boundary(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run_number) + '/em_0000.h5', shape, frame_num) # doesnt mater what boundary is loaded

  # generate em state
  em_state = np.zeros([seq_length] + shape + [frame_num])
  for i in xrange(seq_length):
    em_state[i] = load_em(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run_number) + '/em_' + str(start_index + i).zfill(4) + '.h5', shape, frame_num)

  return em_state, boundary_cond

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
      boundary_cond = load_boundary(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run) + '/em_0000.h5', shape, frame_num)
      boundary_cond = np.float32(boundary_cond)
      boundary_flat = boundary_cond.reshape([np.prod(np.array(shape))])
      boundary_raw = boundary_flat.astype(np.float)

      # save tf records
      ind_dat = 0
      while ind_dat < (num_samples - seq_length - 1):
        seq_frames = np.zeros([seq_length] + shape + [frame_num])
        for i in xrange(seq_length):
          em_state = load_em(FLAGS.data_dir + '/' + dir_name + '/sample_' + str(run) + '/em_' + str(i+ind_dat).zfill(4) + '.h5', shape, frame_num)
          
          em_state = np.float32(em_state)
          seq_frames[i] = em_state 
        overlap = min(4, seq_length)
        ind_dat += seq_length - overlap # overlap between frames

        # make feature map
        feature = make_feature_from_seq(seq_frames, seq_length, shape, frame_num)
        feature['boundary'] = _float_feature(boundary_raw)

        # create example and write it
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

      writer.close()
    
    
