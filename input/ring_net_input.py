
import os
import numpy as np
import tensorflow as tf
#import utils.createTFRecords as createTFRecords
#import systems.balls_createTFRecords as balls_createTFRecords
import systems.fluid_createTFRecords as fluid_createTFRecords
import systems.em_createTFRecords as em_createTFRecords
#import systems.diffusion_createTFRecords as diffusion_createTFRecords
from glob import glob as glb
from tqdm import *


FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
tf.app.flags.DEFINE_integer('min_queue_examples', 400,
                           """ min examples to queue up""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 2,
                           """ number of process threads for que runner """)
tf.app.flags.DEFINE_string('data_dir', '/data',
                           """ base dir for all data""")
tf.app.flags.DEFINE_string('tf_data_dir', '../data',
                           """ base dir for saving tf records data""")

def read_data(filename_queue, seq_length, shape, num_frames, color, raw_type='uint8'):
  """ reads data from tfrecord files.

  Args: 
    filename_queue: A que of strings with filenames 

  Returns:
    frames: the frame data in size (batch_size, seq_length, image height, image width, frames)
  """
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image':tf.FixedLenFeature([],tf.string)
    })
  if raw_type == 'uint8': 
    image = tf.decode_raw(features['image'], tf.uint8)
  elif raw_type == 'float32':
    image = tf.decode_raw(features['image'], tf.float32)
  if color:
    image = tf.reshape(image, [seq_length, shape[0], shape[1], num_frames*3])
  else:
    image = tf.reshape(image, [seq_length, shape[0], shape[1], num_frames])
  image = tf.to_float(image) 
  #Display the training images in the visualizer.
  return image

def read_data_fluid(filename_queue, seq_length, shape, num_frames, color):
  """ reads data from tfrecord files.

  Args: 
    filename_queue: A que of strings with filenames 

  Returns:
    frames: the frame data in size (batch_size, seq_length, image height, image width, frames)
  """
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)

  # make feature dict
  feature_dict = {}
  for i in xrange(seq_length):
    feature_dict['flow/frame_' + str(i)] = tf.FixedLenFeature([np.prod(np.array(shape))*num_frames],tf.float32)
  feature_dict['boundary'] = tf.FixedLenFeature([np.prod(np.array(shape))],tf.float32)
  
  features = tf.parse_single_example(
    serialized_example,
    features=feature_dict) 

  flow = []
  for i in xrange(seq_length):
    #flow.append(tf.decode_raw(features['flow/frame_' + str(i)], tf.float32))
    flow.append(features['flow/frame_' + str(i)])
  #boundary = tf.decode_raw(features['boundary'], tf.float32)
  boundary = features['boundary']

  # reshape
  flow = tf.stack(flow)
  flow = tf.reshape(flow, [seq_length] + shape + [num_frames])
  flow = tf.to_float(flow)
  boundary = tf.reshape(boundary, [1] + shape + [1]) 
  boundary = tf.to_float(boundary)

  return flow, boundary

def read_data_em(filename_queue, seq_length, shape, num_frames, color):
  """ reads data from tfrecord files.

  Args: 
    filename_queue: A que of strings with filenames 

  Returns:
    frames: the frame data in size (batch_size, seq_length, image height, image width, frames)
  """
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

  em = []
  for i in xrange(seq_length):
    #em.append(tf.decode_raw(features['em/frame_' + str(i)], tf.float32))
    em.append(features['em/frame_' + str(i)])
  #boundary = tf.decode_raw(features['boundary'], tf.float32)
  boundary = features['boundary']

  # reshape
  em = tf.stack(em)
  em = tf.reshape(em, [seq_length] + shape + [num_frames])
  em = tf.to_float(em)
  boundary = tf.reshape(boundary, [1] + shape + [1]) 
  boundary = tf.to_float(boundary)

  return em, boundary


def _generate_image_label_batch(image, batch_size):
  """Construct a queued batch of images.
  Args:
    image: 4-D Tensor of [seq, height, width, frame_num] 
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 5D tensor of [batch_size, seq_lenght, height, width, frame_num] size.
  """

  num_preprocess_threads = 1
  if FLAGS.train:
    #Create a queue that shuffles the examples, and then
    #read 'batch_size' images + labels from the example queue.
    frames = tf.train.shuffle_batch(
      [image],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.min_queue_examples + 3 * batch_size,
      min_after_dequeue=FLAGS.min_queue_examples)
  else:
     frames = tf.train.batch(
      [image],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=3 * batch_size)
  return frames

def _generate_image_label_batch_fluid(flow, boundary, batch_size):
  """Construct a queued batch of images.
  Args:
    image: 4-D Tensor of [seq, height, width, frame_num] 
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 5D tensor of [batch_size, seq_lenght, height, width, frame_num] size.
  """

  num_preprocess_threads = FLAGS.num_preprocess_threads
  if FLAGS.train:
    #Create a queue that shuffles the examples, and then
    #read 'batch_size' images + labels from the example queue.
    flows, boundarys = tf.train.shuffle_batch(
      [flow, boundary],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.min_queue_examples + 3 * batch_size,
      min_after_dequeue=FLAGS.min_queue_examples)
  else:
     flows, boundarys = tf.train.batch(
      [flow, boundary],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=3 * batch_size)
  return flows, boundarys

def _generate_image_label_batch_em(em, boundary, batch_size):
  """Construct a queued batch of images.
  Args:
    image: 4-D Tensor of [seq, height, width, frame_num] 
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 5D tensor of [batch_size, seq_lenght, height, width, frame_num] size.
  """

  num_preprocess_threads = FLAGS.num_preprocess_threads
  if FLAGS.train:
    #Create a queue that shuffles the examples, and then
    #read 'batch_size' images + labels from the example queue.
    ems, boundarys = tf.train.shuffle_batch(
      [em, boundary],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.min_queue_examples + 3 * batch_size,
      min_after_dequeue=FLAGS.min_queue_examples)
  else:
     ems, boundarys = tf.train.batch(
      [em, boundary],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=3 * batch_size)
  return ems, boundarys


def fluid_inputs(batch_size, seq_length, shape, num_frames, train=True):
  """Construct cannon input for ring net. just a 28x28 frame video of a bouncing ball 
  Args:
    batch_size: Number of images per batch.
    seq_length: seq of inputs.
  Returns:
    images: Images. 4D tensor. Possible of size [batch_size, 28x28x4].
  """
  # num tf records
  if train:
    #run_num = 100
    run_num = 5
    #run_num = 35
  else:
    run_num = 1

  # make dir name based on shape of simulation 
  dir_name = 'fluid_flow_' + str(shape[0]) + 'x' + str(shape[1])
  if len(shape) > 2:
    dir_name = dir_name + 'x' + str(shape[2])

  dir_name = dir_name + '_'

  if not train:
    dir_name = dir_name + '_test'
 
  print("begining to generate tf records")
  fluid_createTFRecords.generate_tfrecords(seq_length, run_num, shape, num_frames, dir_name)

  tfrecord_filename = glb(FLAGS.tf_data_dir + '/tfrecords/' + str(dir_name) + '/*_seq_length_' + str(seq_length) + '.tfrecords')
 
  filename_queue = tf.train.string_input_producer(tfrecord_filename)

  flow, boundary = read_data_fluid(filename_queue, seq_length, shape, num_frames, False)

  flows, boundarys = _generate_image_label_batch_fluid(flow, boundary, batch_size)

  return flows, boundarys

def em_inputs(batch_size, seq_length, shape, num_frames, train=True):
  """Construct cannon input for ring net. just a 28x28 frame video of a bouncing ball 
  Args:
    batch_size: Number of images per batch.
    seq_length: seq of inputs.
  Returns:
    images: Images. 4D tensor. Possible of size [batch_size, 28x28x4].
  """
  # num tf records
  if train:
    run_num = 10
  else:
    run_num = 1

  # make dir name based on shape of simulation 
  dir_name = 'em_' + str(shape[0]) + 'x' + str(shape[1])
  if len(shape) > 2:
    dir_name = dir_name + 'x' + str(shape[2])

  dir_name = dir_name + '_'

  if not train:
    dir_name = dir_name + '_test'
 
  print("begining to generate tf records")
  em_createTFRecords.generate_tfrecords(seq_length, run_num, shape, num_frames, dir_name)

  tfrecord_filename = glb(FLAGS.tf_data_dir + '/tfrecords/' + str(dir_name) + '/*_seq_length_' + str(seq_length) + '.tfrecords')
 
  filename_queue = tf.train.string_input_producer(tfrecord_filename)

  em, boundary = read_data_em(filename_queue, seq_length, shape, num_frames, False)

  ems, boundarys = _generate_image_label_batch_em(em, boundary, batch_size)

  return ems, boundarys



