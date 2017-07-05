
import os
import numpy as np
import tensorflow as tf
import systems.fluid_createTFRecords as fluid_createTFRecords
import systems.em_createTFRecords as em_createTFRecords
from glob import glob as glb
from tqdm import *

FLAGS = tf.app.flags.FLAGS

# Constants describing the input pipline.
tf.app.flags.DEFINE_integer('min_queue_examples', 400,
                           """ min examples to queue up""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 2,
                           """ number of process threads for que runner """)
tf.app.flags.DEFINE_string('data_dir', '/data',
                           """ base dir for all data""")
tf.app.flags.DEFINE_string('tf_data_dir', '../data',
                           """ base dir for saving tf records data""")
tf.app.flags.DEFINE_integer('tf_seq_length', 30,
                           """ seq length of tf saved records """)


def lat_distortions(lat, distortions):
  if   len(lat.get_shape()) == 5:
    lat = tf.cond(distortions[0]>0.50, lambda: tf.reverse(lat, axis=[2]), lambda: lat)
  elif len(lat.get_shape()) == 6:
    lat = tf.cond(distortions[0]>0.50, lambda: tf.reverse(lat, axis=[2]), lambda: lat)
    lat = tf.cond(0.75<distortions[0], lambda: tf.reverse(lat, axis=[3]), lambda: lat)
    lat = tf.cond(distortions[0]<0.25, lambda: tf.reverse(lat, axis=[3]), lambda: lat)
  return lat

def read_data_fluid(filename_queue, seq_length, shape, num_frames):
  # make reader
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)

  # make feature dict
  feature_dict = {}
  for i in xrange(FLAGS.tf_seq_length):
    feature_dict['flow/frame_' + str(i)] = tf.FixedLenFeature([np.prod(np.array(shape))*num_frames],tf.float32)
  feature_dict['boundary'] = tf.FixedLenFeature([np.prod(np.array(shape))],tf.float32)
  features = tf.parse_single_example(
    serialized_example,
    features=feature_dict) 

  # read seq from record
  seq_of_flow = []
  seq_of_boundary = []
  for sq in xrange(FLAGS.tf_seq_length - seq_length):
    flow = []
    for i in xrange(seq_length):
      flow.append(features['flow/frame_' + str(i+sq)])
    boundary = features['boundary']
    # reshape it
    flow = tf.stack(flow)
    flow = tf.reshape(flow, [seq_length] + shape + [num_frames])
    flow = tf.to_float(flow)
    boundary = tf.reshape(boundary, [1] + shape + [1]) 
    boundary = tf.to_float(boundary)
    seq_of_flow.append(flow)
    seq_of_boundary.append(boundary)
  seq_of_flow = tf.stack(seq_of_flow)
  seq_of_boundary = tf.stack(seq_of_boundary)
   
  return seq_of_flow, seq_of_boundary

def read_data_em(filename_queue, seq_length, shape, num_frames):
  # make reader
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)

  # make feature dict
  feature_dict = {}
  for i in xrange(seq_length):
    feature_dict['em/frame_' + str(i)] = tf.FixedLenFeature([np.prod(np.array(shape))*num_frames],tf.float32)
  feature_dict['boundary'] = tf.FixedLenFeature([np.prod(np.array(shape))],tf.float32)
  features = tf.parse_single_example(
    serialized_example,
    features=feature_dict) 

  # read seq from record
  seq_of_em = []
  seq_of_boundary = []
  for sq in xrange(FLAGS.tf_seq_length - seq_length):
    em = []
    for i in xrange(seq_length):
      em.append(features['em/frame_' + str(i)])
    boundary = features['boundary']
    # reshape it
    em = tf.stack(em)
    em = tf.reshape(em, [seq_length] + shape + [num_frames])
    em = tf.to_float(em)
    boundary = tf.reshape(boundary, [1] + shape + [1]) 
    boundary = tf.to_float(boundary)
    seq_of_em.append(em)
    seq_of_boundary.append(boundary)
  seq_of_em = tf.stack(seq_of_em)
  seq_of_boundary = tf.stack(seq_of_boundary)

  return seq_of_em, seq_of_boundary

def _generate_fluid_batch(seq_of_flow, seq_of_boundary, batch_size):
  num_preprocess_threads = FLAGS.num_preprocess_threads
  flows, boundarys = tf.train.shuffle_batch(
    [seq_of_flow, seq_of_boundary],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    enqueue_many=True,
    capacity=FLAGS.min_queue_examples + 3 * batch_size,
    min_after_dequeue=FLAGS.min_queue_examples)
  return flows, boundarys

def _generate_em_batch(seq_of_em, seq_of_boundary, batch_size):
  num_preprocess_threads = FLAGS.num_preprocess_threads
  ems, boundarys = tf.train.shuffle_batch(
    [seq_of_em, seq_of_boundary],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    enqueue_many=True,
    capacity=FLAGS.min_queue_examples + 3 * batch_size,
    min_after_dequeue=FLAGS.min_queue_examples)
  return ems, boundarys

def fluid_inputs(batch_size, seq_length, shape, num_frames, train=True):
  # number of train simulations
  run_num = 1

  # make dir name based on shape of simulation 
  dir_name = 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1])
  if len(shape) > 2:
    dir_name = dir_name + 'x' + str(shape[2])
  dir_name = dir_name + '_'
 
  print("begining to generate tf records")
  fluid_createTFRecords.generate_tfrecords(FLAGS.tf_seq_length, run_num, shape, num_frames, dir_name)

  # get tfrecord files
  tfrecord_filename = glb(FLAGS.tf_data_dir + '/tfrecords/' + str(dir_name) + '/*_seq_length_' + str(FLAGS.tf_seq_length) + '.tfrecords')

  # make filename que 
  filename_queue = tf.train.string_input_producer(tfrecord_filename)

  # read tfrecords
  seq_of_flow, seq_of_boundary = read_data_fluid(filename_queue, seq_length, shape, num_frames)

  # flip flow as a distortion
  distortions = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)
  seq_of_flow     = lat_distortions(seq_of_flow,     distortions)
  seq_of_boundary = lat_distortions(seq_of_boundary, distortions)

  # construct batch of flows
  flows, boundarys = _generate_fluid_batch(seq_of_flow, seq_of_boundary, batch_size)

  return flows, boundarys

def em_inputs(batch_size, seq_length, shape, num_frames, train=True):
  # number of train simulations
  run_num = 50

  # make dir name based on shape of simulation 
  dir_name = 'em_' + str(shape[0]) + 'x' + str(shape[1])
  if len(shape) > 2:
    dir_name = dir_name + 'x' + str(shape[2])
  dir_name = dir_name + '_'
 
  print("begining to generate tf records")
  em_createTFRecords.generate_tfrecords(FLAGS.tf_seq_length, run_num, shape, num_frames, dir_name)

  # get tfrecord files
  tfrecord_filename = glb(FLAGS.tf_data_dir + '/tfrecords/' + str(dir_name) + '/*_seq_length_' + str(FLAGS.tf_seq_length) + '.tfrecords')
 
  # make filename que 
  filename_queue = tf.train.string_input_producer(tfrecord_filename)

  # read tfrecords
  seq_of_em, seq_of_boundary = read_data_em(filename_queue, seq_length, shape, num_frames)

  # flip flow as a distortion
  distortions = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)
  seq_of_em       = lat_distortions(seq_of_em,       distortions)
  seq_of_boundary = lat_distortions(seq_of_boundary, distortions)

  # construct batch of em
  ems, boundarys = _generate_em_batch(seq_of_em, seq_of_boundary, batch_size)

  return ems, boundarys
