
"""functions used to construct different architectures  
"""

import tensorflow as tf
import numpy as np
import nn 

FLAGS = tf.app.flags.FLAGS

def encoding(inputs, nonlinearity=nn.concat_elu, multi_resolution=True, filter_size=32, keep_p=1.0, gated=True,):
  """Builds encoding part of ring net.
  Args:
    inputs: input to encoder
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  x_i = inputs

  if FLAGS.multi_resolution:
    skip_connections = []
  for i in xrange(FLAGS.nr_downsamples):

    filter_size = FLAGS.filter_size*(2^(i))
    print("filter size for layer " + str(i) + " of encoding is " + str(filter_size))

    x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=2, FLAGS.gated, name="resnet_down_sampled_" + str(i) + "_nr_residual_0") 


    for j in xrange(FLAGS.nr_residual - 1):
      x_i = res_block(x_i, filter_size=FLAGS.filter_size*(2^(i)), nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=1, FLAGS.gated, name="resnet_down_sampled_" + str(i) + "_nr_residual_" + str(j+1))

    if FLAGS.mulit_resolution:
      skip_connections.append(x_i)

  if FLAGS.multi_resolution:
    return skip_connections
  else:
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

  if FLAGS.multi_resolution:
    y_i_store = []
    for i in xrange(FLAGS.nr_downsamples):
      y_i_j = y_i[i]
      for j in xrange(FLAGS.nr_residual_compression):
        y_i_j = res_block(y_i_j, filter_size=int(y_i_j.get_shape()[3]), nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, name="resnet_downsampled_" + str(i) + "_resnet_" + str(j))
      y_i_store.append(y_i_j)
    y_i = y_i_store 

  else:
    for i in xrange(FLAGS.nr_residual_compression):
      y_i = res_block(y_i, filter_size=int(y_i.get_shape()[3]), nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, name="_resnet_" + str(i))

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
  if FLAGS.multi_resolution:
    y_i = y[-1]
  else:
    y_i = y
 
  nonlinearity = set_nonlinearity(FLAGS.nonlinearity)

  for i in xrange(FLAGS.nr_downsamples-1):
    filter_size = FLAGS.filter_size*(2^(FLAGS.nr_downsamples-i-2))
    print("filter size for layer " + str(i) + " of encoding is " + str(filter_size))

    if i != 0 and FLAGS.multi_resolution_skip:
      y_i = res_block(y_i, a=y[-1-i], filter_size=4*filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=1, FLAGS.gated, name="resnet_up_sampled_" + str(i) + "_nr_residual_0")
    else:
      y_i = res_block(y_i, filter_size=4*filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=1, FLAGS.gated, name="resnet_up_sampled_" + str(i) + "_nr_residual_0")
    y_i = PS(y_i, 2, filter_size)


    for j in xrange(FLAGS.nr_residual - 1):
      y_i = res_block(y_i, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p, stride=1, FLAGS.gated, name="resnet_" + str(i) + "_nr_residual_" + str(j+1))

  return y_i

def add_z(y, z):

  if FLAGS.multi_resolution:
    y_new = []
    for i in xrange(FLAGS.nr_downsamples):
      y_shape = int_shape(y[i]) 
      z = fc_layer(z, y_shape[1]*y_shape[2], "fc_z_" + str(i))
      z = tf.reshape(z, [-1, y_shape[1], y_shape[2], 1])
      z = conv_layer(z, 3, 1, y_shape[3], "conv_z_" + str(i))
      y_new.append(y[i] + z)
  else:
    y_shape = int_shape(y]) 
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
      filter_size = FLAGS.filter_size_discriminator*(2^(i))
      print("filter size for discriminator layer " + str(i) + " of encoding is " + str(filter_size))
      x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p_discriminator, stride=2, FLAGS.gated, name="discriminator_" + str(split) + "_resnet_discriminator_down_sampled_" + str(i) + "_nr_residual_0") 
      for j in xrange(FLAGS.nr_residual - 1):
        x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=FLAGS.keep_p_discriminator, stride=1, FLAGS.gated, name="discriminator_" + str(split) + "_resnet_discriminator_" + str(i) + "_nr_residual_" + str(j+1))
  
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

def encode_compress_decode(state, boundry, hidden_state=None, z=None):
 
  state = add_boundry(state, boundry)
 
  y_1 = encoding(state)
  if FLAGS.lstm:
    y_2, hidden_state = lstm_compression(y_1, hidden_state)
  else:
    y_2 = compression(y_1)
  if FLAGS.gan:
    y_2 = add_z(y_2, z)
  x_2 = decoding(y_2) 

  state = apply_boundry(state, boundry)

  return x_2, hidden_state




