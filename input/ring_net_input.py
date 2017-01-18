
import os
import numpy as np
import tensorflow as tf
import utils.createTFRecords as createTFRecords
import systems.balls_createTFRecords as balls_createTFRecords
import systems.fluid_createTFRecords as fluid_createTFRecords
import systems.diffusion_createTFRecords as diffusion_createTFRecords
from glob import glob as glb
from tqdm import *


FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
tf.app.flags.DEFINE_integer('min_queue_examples', 300,
                           """ min examples to queue up""")

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
  features = tf.parse_single_example(
    serialized_example,
    features={
      'flow':tf.FixedLenFeature([],tf.string),
      'boundry':tf.FixedLenFeature([],tf.string)
    }) 
  flow = tf.decode_raw(features['flow'], tf.float32)
  boundry = tf.decode_raw(features['boundry'], tf.float32)

  # reshape
  flow = tf.reshape(flow, [seq_length, shape[0], shape[1], num_frames])
  flow = tf.to_float(flow)
  boundry = tf.reshape(boundry, [1, shape[0], shape[1], 2]) 
  boundry = tf.to_float(boundry)
  boundry = tf.concat(0, [boundry]*seq_length)
  print(boundry.get_shape())
  #Display the training images in the visualizer.
  return flow, boundry

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

def _generate_image_label_batch_fluid(flow, boundry, batch_size):
  """Construct a queued batch of images.
  Args:
    image: 4-D Tensor of [seq, height, width, frame_num] 
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 5D tensor of [batch_size, seq_lenght, height, width, frame_num] size.
  """

  num_preprocess_threads = 2
  if FLAGS.train:
    #Create a queue that shuffles the examples, and then
    #read 'batch_size' images + labels from the example queue.
    flows, boundrys = tf.train.shuffle_batch(
      [flow, boundry],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.min_queue_examples + 3 * batch_size,
      min_after_dequeue=FLAGS.min_queue_examples)
  else:
     flows, boundrys = tf.train.batch(
      [flow, boundry],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=3 * batch_size)
  return flows, boundrys


def video_inputs(batch_size, seq_length):
  """Construct video input for ring net. given a video_dir that contains videos this will check to see if there already exists tf recods and makes them. Then returns batchs
  Args:
    batch_size: Number of images per batch.
    seq_length: seq of inputs.
  Returns:
    images: Images. 4D tensor. Possible of size [batch_size, 84x84x4].
  """
  params = None
  params_loss = None

  # get list of video file names
  video_filename = glb(FLAGS.data_dir + 'videos/'+FLAGS.video_dir+'/*') 

  if FLAGS.model in ("fully_connected_84x84x4", "lstm_84x84x4"):
    shape = (84,84)
    num_frames = 4
    color = False
  if FLAGS.model in ("fully_connected_84x84x3", "lstm_84x84x3"):
    shape = (84, 84)
    num_frames = 1 
    color = True

  print("begining to generate tf records")
  for f in video_filename:
    createTFRecords.generate_tfrecords(f, seq_length, shape, num_frames, color)
 
  # get list of tfrecords 
  tfrecord_filename = glb('../data/tfrecords/'+FLAGS.video_dir+'/*seq_' + str(seq_length) + '_size_' + str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(num_frames) + '_color_' + str(color) + '.tfrecords') 
  
  
  filename_queue = tf.train.string_input_producer(tfrecord_filename) 

  image = read_data(filename_queue, seq_length, shape, num_frames, color)
  
  if color:
    display_image = tf.split(3, 3, image)
    tf.image_summary('images', display_image[0])
  else:
    tf.image_summary('images', image)

  image = tf.div(image, 255.0) 

  frames = _generate_image_label_batch(image, batch_size)
 
  return frames

def balls_inputs(batch_size, seq_length):
  """Construct cannon input for ring net. just a 28x28 frame video of a bouncing ball 
  Args:
    batch_size: Number of images per batch.
    seq_length: seq of inputs.
  Returns:
    images: Images. 4D tensor. Possible of size [batch_size, 28x28x4].
  """
  params = None
  params_loss = None
 
  # num samples per tfrecord 
  num_samples = 1000
  # num tf records
  if FLAGS.train == True:
    run_num = 100
  else:
    run_num = 5
 
  dir_name = 'balls'
  if not FLAGS.train:
    dir_name = dir_name + '_test'
 
  print("begining to generate tf records")
  for i in tqdm(xrange(run_num)): 
    balls_createTFRecords.generate_tfrecords(i, num_samples, seq_length, dir_name)
    
  tfrecord_filename = glb(FLAGS.data_dir + 'tfrecords/' + dir_name + '/*num_samples_' + str(num_samples) + '_seq_length_' + str(seq_length) + '_friction_' + str(FLAGS.friction) + '_num_balls_' + str(FLAGS.num_balls) + '.tfrecords') 
 
  filename_queue = tf.train.string_input_producer(tfrecord_filename) 

  image = read_data(filename_queue, seq_length, (32, 32), 1, True, 'float32')
  tf.image_summary('images', image)
  
  #image = tf.div(image, 255.0) 

  frames = _generate_image_label_batch(image, batch_size)

  return frames

def diffusion_inputs(batch_size, seq_length):
  """Construct cannon input for ring net. just a 28x28 frame video of a bouncing ball 
  Args:
    batch_size: Number of images per batch.
    seq_length: seq of inputs.
  Returns:
    images: Images. 4D tensor. Possible of size [batch_size, 28x28x4].
  """
  # num tf records
  if FLAGS.train == True:
    run_num = 1000
  else:
    run_num = 100 

  shape=(32,32)
 
  dir_name = 'diffusion'
  if not FLAGS.train:
    dir_name = dir_name + '_test'

  diffusion_createTFRecords.generate_tfrecords(seq_length, run_num, dir_name)

  tfrecord_filename = glb(FLAGS.data_dir + 'tfrecords/' + dir_name + '/*_seq_length_' + str(seq_length) + '.tfrecords')
 
  filename_queue = tf.train.string_input_producer(tfrecord_filename)

  image = read_data(filename_queue, seq_length, shape, 1, False, 'float32')
  tf.image_summary('images', image[:,:,:,:])
  
  frames = _generate_image_label_batch(image, batch_size)

  return frames

def fluid_inputs(batch_size, seq_length):
  """Construct cannon input for ring net. just a 28x28 frame video of a bouncing ball 
  Args:
    batch_size: Number of images per batch.
    seq_length: seq of inputs.
  Returns:
    images: Images. 4D tensor. Possible of size [batch_size, 28x28x4].
  """
  # num tf records
  if FLAGS.train == True:
    run_num = 50
  else:
    run_num = 1

  shape=(401,101)
  num_frames = 2

  dir_name = 'fluid_flow'
  if not FLAGS.train:
    dir_name = dir_name + '_test'
 
  fluid_createTFRecords.generate_tfrecords(seq_length, run_num, shape, num_frames, dir_name)

  tfrecord_filename = glb(FLAGS.data_dir + 'tfrecords/' + str(dir_name) + '/*_seq_length_' + str(seq_length) + '.tfrecords')
 
  filename_queue = tf.train.string_input_producer(tfrecord_filename)

  flow, boundry = read_data_fluid(filename_queue, seq_length, shape, num_frames, False)
  #
  tf.image_summary('x', flow[:,:,:,0:1])
  tf.image_summary('y', flow[:,:,:,1:2])
  tf.image_summary('boundry', boundry[:,:,:,0:1])

  #image = tf.div(image, 255.0) 

  flows, boundrys = _generate_image_label_batch_fluid(flow, boundry, batch_size)

  return flows, boundrys

