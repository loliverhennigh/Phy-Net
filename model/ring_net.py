
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
from nn import *
import input.ring_net_input as ring_net_input

FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.

################# system params
tf.app.flags.DEFINE_string('system', 'fluid_flow',
                           """ system to compress """)
tf.app.flags.DEFINE_integer('lattice_size', 3,
                           """ size of lattice """)
tf.app.flags.DEFINE_string('dimensions', '256x256',
                           """ dimension of simulation with x between value """)

################# running params
tf.app.flags.DEFINE_string('base_dir', '../checkpoints',
                            """dir to store trained net """)
tf.app.flags.DEFINE_bool('restore', True,
                            """ restore model if there is one """)

################# model params
## resnet params
tf.app.flags.DEFINE_integer('nr_residual', 2,
                           """ number of residual blocks before down sizing """)
tf.app.flags.DEFINE_integer('nr_downsamples', 3,
                           """ numper of downsamples """)
tf.app.flags.DEFINE_string('nonlinearity', "concat_elu",
                           """ what nonlinearity to use, leakey_relu, relu, elu, concat_elu """)
tf.app.flags.DEFINE_float('keep_p', 1.0,
                           """ keep probability for res blocks """)
tf.app.flags.DEFINE_bool('gated', False,
                           """ gated res blocks """)
tf.app.flags.DEFINE_integer('filter_size', 16,
                           """ filter size for first res block. the rest of the filters are 2x every downsample """)
## lstm params
tf.app.flags.DEFINE_bool('lstm', False,
                           """ lstm or non recurrent""")
tf.app.flags.DEFINE_integer('nr_residual_compression', 4,
                           """ number of residual compression layers """)
tf.app.flags.DEFINE_integer('filter_size_compression', 64,
                           """ filter size for compression piece """)
## gan params
tf.app.flags.DEFINE_bool('gan', False,
                           """ use gan training """)
tf.app.flags.DEFINE_integer('nr_discriminators', 1,
                           """ number of discriminators to train """)
tf.app.flags.DEFINE_integer('z_size', 50,
                           """ size of z vector """)
tf.app.flags.DEFINE_integer('nr_residual_discriminator', 1,
                           """ number of residual blocks before down sizing """)
tf.app.flags.DEFINE_integer('nr_downsamples_discriminator', 3,
                           """ numper of downsamples """)
tf.app.flags.DEFINE_float('keep_p_discriminator', 1.0,
                           """ keep probability for res blocks """)
tf.app.flags.DEFINE_integer('filter_size_discriminator', 32,
                           """ filter size for first res block of discriminator """)
tf.app.flags.DEFINE_integer('lstm_size_discriminator', 512,
                           """ size of lstm cell in discriminator """)
 


################# optimize params
tf.app.flags.DEFINE_string('optimizer', "adam",
                           """ what optimizer to use """)
tf.app.flags.DEFINE_float('reconstruction_lr', 1e-5,
                           """ learning rete for reconstruction """)
tf.app.flags.DEFINE_float('gan_lr', 2e-5,
                           """ learning rate for training gan """)
tf.app.flags.DEFINE_float('lambda_reconstruction', 1.0,
                           """ weight of reconstruction error """) 
tf.app.flags.DEFINE_float('lambda_divergence', 0.2,
                           """ weight of divergence error """)

################# train params
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """ max steps to train """)
tf.app.flags.DEFINE_integer('unroll_length', 5,
                           """ unroll length """)
tf.app.flags.DEFINE_integer('init_unroll_length', 0,
                           """ inital unroll length before training """)
tf.app.flags.DEFINE_bool('unroll_from_true', False,
                           """ use the true data when unrolling the network (probably just used for unroll_length 1 when doing curriculum learning""")
tf.app.flags.DEFINE_integer('batch_size', 4,
                           """ batch size """)

################# test params
tf.app.flags.DEFINE_string('test_dimensions', '1024x1024',
                           """ test video dimentions """)
tf.app.flags.DEFINE_integer('video_length', 50,
                           """ video dimentions """)
tf.app.flags.DEFINE_integer('test_length', 50,
                           """ sequence length for testing (making error plots) """)
tf.app.flags.DEFINE_integer('test_nr_runs', 50,
                           """ number of simulations to test on (making error plots) """)
tf.app.flags.DEFINE_integer('test_nr_per_simulation', 300,
                           """ number of test runs per simulations (making error plots) """)

################# input params
tf.app.flags.DEFINE_bool('train', True,
                           """ train or test """)

def inputs(empty=False, shape=None):
  """makes input vector
  Args:
    empty will just return an empty state to fill with a feed dict
  Return:
    x: input vector, may be filled 
  """
  if shape is None:
    shape = FLAGS.dimensions.split('x')
    shape = map(int, shape)
  frame_num = FLAGS.lattice_size # 3 for 2D and 4 for 3D (this will change with mag stuff)
  if empty:
    state = tf.placeholder(tf.float32, [1] + shape + [frame_num])
    boundary = tf.placeholder(tf.float32, [1] + shape + [1])
  elif FLAGS.system == "fluid_flow":
    state, boundary = ring_net_input.fluid_inputs(FLAGS.batch_size, FLAGS.init_unroll_length + FLAGS.unroll_length, shape, frame_num, FLAGS.train)
 
  if FLAGS.gan:
    z = tf.placeholder("float", [None, total_unroll_length, FLAGS.z_size])
    return state, boundary, z
  else:
    return state, boundary

def encoding(inputs, name='', boundary=False):
  """Builds encoding part of ring net.
  Args:
    inputs: input to encoder
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  x_i = inputs

  nonlinearity = set_nonlinearity(FLAGS.nonlinearity)

  for i in xrange(FLAGS.nr_downsamples):

    filter_size = FLAGS.filter_size*(pow(2,i))
    #print("filter size for layer " + str(i) + " of encoding is " + str(filter_size))

    x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=2, gated=FLAGS.gated, name=name + "resnet_down_sampled_" + str(i) + "_nr_residual_0") 


    for j in xrange(FLAGS.nr_residual - 1):
      x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=1, gated=FLAGS.gated, name=name + "resnet_down_sampled_" + str(i) + "_nr_residual_" + str(j+1))

  if boundary:
    x_i = res_block(x_i, filter_size=FLAGS.filter_size_compression*2, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=1, gated=FLAGS.gated, name=name + "resnet_last_before_compression")
  else:
    x_i = res_block(x_i, filter_size=FLAGS.filter_size_compression, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=1, gated=FLAGS.gated, name=name + "resnet_last_before_compression")

  return x_i

def compression(y):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2

  y_i = y

  nonlinearity = set_nonlinearity(FLAGS.nonlinearity)

  for i in xrange(FLAGS.nr_residual_compression):
    #print("resnet compression " + str(i))
    y_i = res_block(y_i, filter_size=FLAGS.filter_size_compression, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=1, gated=FLAGS.gated, name="resnet_compression_" + str(i))

  return y_i

# not functional yet!!!
def compression_lstm(y, hidden_state=None):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2

  y_i = y

  if hidden_state is not None:
    hidden_state_1_i = hidden_state[0] 
    hidden_state_2_i = hidden_state[1]

  hidden_state_1_i_new = []
  hidden_state_2_i_new = []

  if FLAGS.multi_resolution:
    for i in xrange(FLAGS.nr_downsamples):
      hidden_state_1_i_j_new = []
      hidden_state_2_i_j_new = []
      y_i_new = []
      for j in xrange(FLAGS.nr_residual_compression):
        if hidden is not None:
          y_i, hidden_state_1_store, hidden_state_2_store = res_block_lstm(y_i, hidden_state_1_i[i][j], hidden_state_2_i[i][j], FLAGS.keep_p, name="resnet_downsampled_" + str(i) + "_resnet_lstm_" + str(j))
        else:
          y_i, hidden_state_1_store, hidden_state_2_store = res_block_lstm(y_i, None, None, FLAGS.keep_p, name="resnet_downsampled_" + str(i) + "_resnet_lstm_" + str(j))
        hidden_state_1_i_j_new.append(hidden_state_1_store)
        hidden_state_2_i_j_new.append(hidden_state_2_store)
      hidden_state_1_i_new.append(hidden_state_1_i_j_new) 
      hidden_state_2_i_new.append(hidden_state_2_i_j_new) 

  else:
    for i in xrange(FLAGS.nr_residual_compression):
      if hidden is not None:
        y_i, hidden_state_1_store, hidden_state_2_store = res_block_lstm(y_i, hidden_state_1_i[i], hidden_state_2_i[i], FLAGS.keep_p, name="resnet_lstm_" + str(i))
      else:
        y_i, hidden_state_1_store, hidden_state_2_store = res_block_lstm(y_i, None, None, FLAGS.keep_p, name="resnet_lstm_" + str(i))
      hidden_state_1_i_new.append(hidden_state_1_store)
      hidden_state_2_i_new.append(hidden_state_2_store)

  hidden = [hidden_state_1_i_new, hidden_state_2_i_new]

  return y_i, hidden 

def decoding(y):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_2 -> x_2
  y_i = y
 
  nonlinearity = set_nonlinearity(FLAGS.nonlinearity)

  for i in xrange(FLAGS.nr_downsamples-1):
    filter_size = FLAGS.filter_size*pow(2,FLAGS.nr_downsamples-i-2)
    #print("decoding filter size for layer " + str(i) + " of encoding is " + str(filter_size))
    y_i = transpose_conv_layer(y_i, 3, 2, filter_size, "up_conv_" + str(i))

    for j in xrange(FLAGS.nr_residual):
      y_i = res_block(y_i, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=1, gated=FLAGS.gated, name="resnet_up_sampled_" + str(i) + "_nr_residual_" + str(j+1))

  y_i = transpose_conv_layer(y_i, 3, 2, FLAGS.lattice_size, "up_conv_" + str(FLAGS.nr_downsamples))

  return tf.nn.tanh(y_i)

def add_z(y, z):
  y_shape = int_shape(y) 
  z = fc_layer(z, y_shape[1]*y_shape[2], "fc_z_" + str(i))
  z = tf.reshape(z, [-1, y_shape[1], y_shape[2], 1])
  z = conv_layer(z, 3, 1, y_shape[3], "conv_z_" + str(i))
  y_new = y + z

  return y_new

def discriminator(output, hidden_state=None):

  x_i = output

  nonlinearity = set_nonlinearity(FLAGS.nonlinearity)

  label = []

  for split in xrange(FLAGS.nr_discriminators):
    for i in xrange(FLAGS.nr_downsamples):
      filter_size = FLAGS.filter_size_discriminator*pow(2,i)
      #print("filter size for discriminator layer " + str(i) + " of encoding is " + str(filter_size))
      x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p_discriminator, stride=2, gated=FLAGS.gated, name="discriminator_" + str(split) + "_resnet_discriminator_down_sampled_" + str(i) + "_nr_residual_0") 
      for j in xrange(FLAGS.nr_residual - 1):
        x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p_discriminator, stride=1, gated=FLAGS.gated, name="discriminator_" + str(split) + "_resnet_discriminator_" + str(i) + "_nr_residual_" + str(j+1))
  
    with tf.variable_scope("discriminator_LSTM_" + str(split), initializer = tf.random_uniform_initializer(-0.01, 0.01)):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.lstm_size_discriminator, forget_bias=1.0)
      if hidden_state == None:
        batch_size = x_i.get_shape()[0]
        hidden_state = lstm_cell.zero_state(batch_size, tf.float32)
  
      x_i, new_state = lstm_cell(x_i, hidden_state)

      x_i = fc_layer(x_i, 1, "discriminator_fc_" + str(split), False, True)
  
      label.append(x_i)

  label = tf.pack(label)

  return label

def encode_compress_decode(state, boundary, hidden_state=None, z=None):
 
  y_1 = encoding(state)
  small_boundary = encoding(boundary, name='boundary')
  y_1 = small_boundary * y_1
  if FLAGS.lstm:
    y_2, hidden_state = lstm_compression(y_1, hidden_state)
  else:
    y_2 = compression(y_1)
  if FLAGS.gan:
    y_2 = add_z(y_2, z)
  x_2 = decoding(y_2) 

  return x_2, hidden_state

def unroll(state, boundary, z=None):

  total_unroll_length = FLAGS.init_unroll_length + FLAGS.unroll_length 
 
  if FLAGS.lstm:
    # need to implement
    exit()
  else:
    # store all out
    x_out = []
    # encode
    y_1 = encoding(state[:,0])
    small_boundary = encoding(boundary[:,0], name='boundry_', boundary=True)
    # apply boundary
    [small_boundary_mul, small_boundary_add] = tf.split(len(small_boundary.get_shape())-1, 2, small_boundary)
    y_1 = (small_boundary_mul * y_1) + small_boundary_add
    # add z if gan training
    if FLAGS.gan:
      y_1 = add_z(y_1, z)
    # unroll all
    for i in xrange(FLAGS.unroll_length):
      # set reuse to true
      if i > 0:
        tf.get_variable_scope().reuse_variables()
        if FLAGS.unroll_from_true: # for testing (no compression at all)
          y_1 = encoding(state[:,i]) 
          y_1 = (small_boundary_mul * y_1) + small_boundary_add
        
      # decode and add to list
      x_2 = decoding(y_1)
      x_out.append(x_2)
      # display
      tf.summary.image('generated_x_' + str(i), x_2[:,:,:,0:1])
      tf.summary.image('generated_y_' + str(i), x_2[:,:,:,1:2])
      tf.summary.image('generated_density_' + str(i), x_2[:,:,:,2:3])
      # compression
      y_1 = compression(y_1)
      # boundary
      y_1 = (small_boundary_mul * y_1) + small_boundary_add
      # add z if gan training
      if FLAGS.gan:
        y_1 = add_z(y_1, z)

  x_out = tf.pack(x_out)
  x_out = tf.transpose(x_out, perm=[1,0,2,3,4])
  return x_out

def continual_unroll(state, boundary, z=None):

  if FLAGS.lstm:
    # need to implement
    exit()
  else:
    # store all out
    y_1 = encoding(state)
    small_boundary = encoding(boundary, name='boundry_', boundary=True)
    # apply boundary
    [small_boundary_mul, small_boundary_add] = tf.split(len(small_boundary.get_shape())-1, 2, small_boundary)
    y_1_boundary = (small_boundary_mul * y_1) + small_boundary_add
    # add z if gan training
    if FLAGS.gan:
      y_1_boundary = add_z(y_1_boundary, z)
    # unroll all
    x_2 = decoding(y_1_boundary)
    y_2 = compression(y_1_boundary)

  return y_1, small_boundary_mul, small_boundary_add, x_2, y_2


  """
  x_2_o = []
  if FLAGS.gan:
    x_2, hidden_state = encode_compress_decode(state[:,0], z=z[:,0], hidden_state=None)
    gan_t_label, gan_t_hidden_state = discriminator(state[:,0], None)
  else: 
    x_2, hidden_state = encode_compress_decode(state[:,0], hidden_state=None)


  tf.get_variable_scope().reuse_variable()
  if FLAGS.gan:
    gan_g_label, gan_g_hidden_state = ring_net.discriminator(state[:,0], None)

  for i in xrange(FLAGS.init_unroll_length-1):
    if FLAGS.gan: 
      # unroll generator network 
      x_2, hidden_state = ring_net.encode_compress_decode(state[:,i+1], hidden_state, z[:,i+1,:])
  
      # unroll discriminator network on true
      gan_t_label, gan_t_hidden_state = ring_net.discriminator(state[:,i+1], gan_t_hidden_state)

      # unroll discriminator network on generated
      gan_g_label, gan_g_hidden_state = ring_net.discriminator(state[:,i+1], gan_g_hidden_state)

    else:
      # unroll generator network 
      x_2, hidden_state = ring_net.encode_compress_decode(flow_boundary[:,i+1], hidden_state)

  x_2_o.append(x_2)

  if FLAGS.gan:
    # unroll discriminator network on true
    gan_t_label, gan_t_hidden_state = ring_net.discriminator(state[:,FLAGS.init_unroll_length], gan_t_hidden_state)

    # unroll discriminator network on generated
    gan_g_label, gan_g_hidden_state = ring_net.discriminator(x_2, gan_g_hidden_state)

  for i in xrange(FLAGS.unroll_length):
    if FLAGS.unroll_from_true:
      next_state = state[:,i+FLAGS.init_unroll_length]
    else:
      next_state = x2

    if FLAGS.gan: 
      # unroll generator network 
      x_2, hidden_state = ring_net.encode_compress_decode(next_state, hidden_state, z[:,i+FLAGS.init_unroll_length,:])
  
      # unroll discriminator network on true
      gan_t_label, gan_t_hidden_state = ring_net.discriminator(state[:,i+FLAGS.init_unroll_length+1], gan_t_hidden_state)

      # unroll discriminator network on generated
      gan_g_label, gan_g_hidden_state = ring_net.discriminator(x_2, gan_g_hidden_state)

    else:
      # unroll generator network 
      x_2, hidden_state = ring_net.encode_compress_decode(next_state, hidden_state)

    x_2_o.append(x_2)

  x_2_o = tf.pack(x_2_o)
  x_2_o = tf.transpose(x_2_o, perm[1,0])

  if FLAGS.gan
    return x_2_o, gan_t_label, gan_g_label
  else:
    return x_2_o
  """

