
"""helper functions to unrap the network for testing.
"""
import tensorflow as tf
import numpy as np
import ring_net 
import architecture

FLAGS = tf.app.flags.FLAGS

def fully_connected_unwrap_generate(inputs, input_seq_length, output_seq_length):
  # first run  
  output_f_mean = []
  output_f_stddev = []
  mean_0, stddev_0 = ring_net.encoding(inputs[:, 0, :, :, :], 1.0) 
  output_f_mean.append(mean_0)
  output_f_stddev.append(stddev_0)

  output_t_mean = []
  output_t_stddev = []
  mean_1, stddev_1 = ring_net.compression(mean_0, stddev_0, 1.0)
  output_t_mean.append(mean_1)
  output_t_stddev.append(stddev_1)

  output_g = []
  x_1 = ring_net.decoding(mean_1, stddev_1, 1.0)
  output_g.append(x_1)

  # set reuse to true
  tf.get_variable_scope().reuse_variables() 

  # fully unwrap 
  for i in xrange(input_seq_length-1):
    # encode
    mean_i, stddev_i = ring_net.encoding(inputs[:,i+1,:,:,:], 1.0)
    output_f_mean.append(mean_i)
    output_f_stddev.append(stddev_i)
  
    # compress
    mean_i_plus, stddev_i_plus = ring_net.compression(mean_i, stddev_i, 1.0)
    output_t_mean.append(mean_i_plus)
    output_t_stddev.append(stddev_i_plus)
 
    # decode
    x_i_plus = ring_net.decoding(mean_i_plus, stddev_i_plus, 1.0)
    output_g.append(x_i_plus)
    tf.image_summary('images_encoding', x_i_plus)

  # now generate new seqs 
  for i in xrange(output_seq_length-1):
    # compress
    mean_i_plus, stddev_i_plus = ring_net.compression(mean_i_plus, stddev_i_plus, 1.0)
    output_t_mean.append(mean_i_plus)
    output_t_stddev.append(stddev_i_plus)
 
    # decode
    x_i_plus = ring_net.decoding(mean_i_plus, stddev_i_plus, 1.0)
    output_g.append(x_i_plus)
    tf.image_summary('images_encoding', x_i_plus)

  # compact outputs
  # f
  output_f_mean = tf.pack(output_f_mean)
  output_f_mean = tf.transpose(output_f_mean, perm=[1,0,2])
  output_f_stddev = tf.pack(output_f_stddev)
  output_f_stddev = tf.transpose(output_f_stddev, perm=[1,0,2])
  # t
  output_t_mean = tf.pack(output_t_mean)
  output_t_mean = tf.transpose(output_t_mean, perm=[1,0,2])
  output_t_stddev = tf.pack(output_t_stddev)
  output_t_stddev = tf.transpose(output_t_stddev, perm=[1,0,2])
  # g
  output_g = tf.pack(output_g)
  output_g = tf.transpose(output_g, perm=[1,0,2,3,4])
  
  return output_f_mean, output_f_stddev, output_t_mean, output_t_stddev, output_g, output_autoencoder 

def lstm_unwrap_generate(inputs, input_seq_length, output_seq_length):
  # first run  
  output_f_mean = []
  output_f_stddev = []
  mean_0, stddev_0 = ring_net.encoding(inputs[:, 0, :, :, :], 1.0) 
  output_f_mean.append(mean_0)
  output_f_stddev.append(stddev_0)

  output_t_mean = []
  output_t_stddev = []
  mean_1, stddev_1, hidden = ring_net.lstm_compression(mean_0, stddev_0, None, 1.0)
  output_t_mean.append(mean_1)
  output_t_stddev.append(stddev_1)

  output_g = []
  x_1 = ring_net.decoding(mean_1, stddev_1, 1.0)
  output_g.append(x_1)

  # set reuse to true
  tf.get_variable_scope().reuse_variables() 

  # fully unwrap 
  for i in xrange(input_seq_length-1):
    # encode
    mean_i, stddev_i = ring_net.encoding(inputs[:,i+1,:,:,:], 1.0)
    output_f_mean.append(mean_i)
    output_f_stddev.append(stddev_i)
  
    # compress
    mean_i_plus, stddev_i_plus, hidden = ring_net.lstm_compression(mean_i, stddev_i, hidden, 1.0)
    output_t_mean.append(mean_i_plus)
    output_t_stddev.append(stddev_i_plus)
 
    # decode
    x_i_plus = ring_net.decoding(mean_i_plus, stddev_i_plus, 1.0)
    output_g.append(x_i_plus)
    tf.image_summary('images_encoding', x_i_plus)

  # now generate new seqs 
  for i in xrange(output_seq_length-1):
    # compress
    mean_i_plus, stddev_i_plus, hidden = ring_net.lstm_compression(mean_i_plus, stddev_i_plus, hidden, 1.0)
    output_t_mean.append(mean_i_plus)
    output_t_stddev.append(stddev_i_plus)
 
    # decode
    x_i_plus = ring_net.decoding(mean_i_plus, stddev_i_plus, 1.0)
    output_g.append(x_i_plus)
    tf.image_summary('images_encoding', x_i_plus)

  # compact outputs
  # f
  output_f_mean = tf.pack(output_f_mean)
  output_f_mean = tf.transpose(output_f_mean, perm=[1,0,2])
  output_f_stddev = tf.pack(output_f_stddev)
  output_f_stddev = tf.transpose(output_f_stddev, perm=[1,0,2])
  # t
  output_t_mean = tf.pack(output_t_mean)
  output_t_mean = tf.transpose(output_t_mean, perm=[1,0,2])
  output_t_stddev = tf.pack(output_t_stddev)
  output_t_stddev = tf.transpose(output_t_stddev, perm=[1,0,2])
  # g
  output_g = tf.pack(output_g)
  output_g = tf.transpose(output_g, perm=[1,0,2,3,4])
  
  return output_f_mean, output_f_stddev, output_t_mean, output_t_stddev, output_g, output_autoencoder 

def encoding(inputs, keep_prob):
  # calc y_0
  mean, stddev = ring_net.encoding(inputs[:, 0, :, :, :], keep_prob)  
  return mean, stddev

def fully_connected_step(y_0, keep_prob):
  # calc x_0
  x_0 = ring_net.decoding(y_0)
 
  # calc next state
  y_1 = ring_net.compression(y_0, keep_prob)

  return x_0, y_1

def lstm_step(y_0, hidden_state, keep_prob):
  # calc x_0
  x_0 = ring_net.decoding(y_0)
  
  # calc next state
  y_1, hidden_state = ring_net.lstm_compression(y_0, hidden_state, keep_prob)

  return x_0, y_1, hidden_state

