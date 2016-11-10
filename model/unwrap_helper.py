
"""helper functions to unrap the network for training.
"""
import tensorflow as tf
import numpy as np
import ring_net 
import architecture

FLAGS = tf.app.flags.FLAGS

def lstm_unwrap(inputs, keep_prob_encoding, keep_prob_lstm, keep_prob_decoding, seq_length, train_piece):
 
  # first run  
  output_f = []
  y_0 = ring_net.encoding(inputs[:, 0, :, :, :],keep_prob_encoding) 
  output_f.append(y_0)

  output_t = []
  y_1, hidden = ring_net.lstm_compression(y_0, None, keep_prob_lstm, encode=True)
  output_t.append(y_1)

  output_g = []
  x_1 = ring_net.decoding(y_1, keep_prob_decoding)
  output_g.append(x_1)
  if FLAGS.model != 'lstm_32x32x1':
    tf.image_summary('images_encode_1', x_1[:,:,:,0:3])
  else:
    tf.image_summary('images_encode_1', x_1[:,:,:,:])

  # set reuse to true
  tf.get_variable_scope().reuse_variables()

  # first get encoding
  for i in xrange(seq_length-1):
    # encode
    y_i = ring_net.encoding(inputs[:,i+1,:,:,:], keep_prob_encoding)
    output_f.append(y_i)
  
    # compress
    if i < 4:
      y_1, hidden = ring_net.lstm_compression(y_i, hidden, keep_prob_lstm, encode=True)
    else:
      y_1, hidden = ring_net.lstm_compression(y_1, hidden, keep_prob_lstm, encode=False)
    
    output_t.append(y_1)
 
    # decode
    x_i_plus = ring_net.decoding(y_1, keep_prob_decoding)
    output_g.append(x_i_plus)
    if FLAGS.model != 'lstm_32x32x1':
      tf.image_summary('images_encoding_' + str(i+2), x_i_plus[:,:,:,0:3])
    else:
      tf.image_summary('images_encoding_' + str(i+2), x_i_plus[:,:,:,:])

  # now do the autoencoding part
  output_autoencoder = []
  for i in xrange(seq_length):
    x_i = ring_net.decoding(output_f[i], keep_prob_decoding)
    output_autoencoder.append(x_i)
    if FLAGS.model != 'lstm_32x32x1':
      tf.image_summary('images_autoencoding_' + str(i+2), x_i[:,:,:,0:3])
    else:
      tf.image_summary('images_autoencoding_' + str(i+2), x_i[:,:,:,:])

  # compact outputs
  # f
  output_f = tf.pack(output_f)
  output_f = tf.transpose(output_f, perm=[1,0,2])
  # t
  output_t = tf.pack(output_t)
  output_t = tf.transpose(output_t, perm=[1,0,2])
  # g
  output_g = tf.pack(output_g)
  output_g = tf.transpose(output_g, perm=[1,0,2,3,4])
  # autoencoder
  output_autoencoder = tf.pack(output_autoencoder)
  output_autoencoder = tf.transpose(output_autoencoder, perm=[1,0,2,3,4])
  
  return output_f, output_t, output_g, output_autoencoder 

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
    mean_1, stddev_1, hidden = ring_net.lstm_compression(mean_i, stddev_i, hidden, 1.0)
    output_t_mean.append(mean_1)
    output_t_stddev.append(stddev_1)
 
    # decode
    x_i_plus = ring_net.decoding(mean_1, stddev_1, 1.0)
    output_g.append(x_i_plus)
    tf.image_summary('images_encoding', x_i_plus[:,:,:,0:3])

  # now generate new seqs 
  for i in xrange(output_seq_length):
    # encode
    mean_i, stddev_i = ring_net.encoding(inputs[:,i+input_seq_length,:,:,:], 1.0)
    output_f_mean.append(mean_i)
    output_f_stddev.append(stddev_i)

    # compress
    mean_1, stddev_1, hidden = ring_net.lstm_compression(mean_1, stddev_1, hidden, 1.0)
    output_t_mean.append(mean_1)
    output_t_stddev.append(stddev_1)
 
    # decode
    x_i_plus = ring_net.decoding(mean_1, stddev_1, 1.0)
    output_g.append(x_i_plus)
    tf.image_summary('images_encoding', x_i_plus[:,:,:,0:3])

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
  
  return output_f_mean, output_f_stddev, output_t_mean, output_t_stddev, output_g

def lstm_unwrap_generate_3_skip(inputs, second_seq, third_seq):
  index = 0

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
  index = index + 1 

  output_g = []
  x_1 = ring_net.decoding(mean_1, stddev_1, 1.0)
  output_g.append(x_1)

  # set reuse to true
  tf.get_variable_scope().reuse_variables() 

  # first unwrap 3 and get images 
  for i in xrange(2):
    # compress 
    mean_1, stddev_1, hidden = ring_net.lstm_compression(mean_1, stddev_1, hidden, 1.0)
    output_t_mean.append(mean_1)
    output_t_stddev.append(stddev_1)
    index = index + 1 
 
    # decode
    x_i_plus = ring_net.decoding(mean_1, stddev_1, 1.0)
    output_g.append(x_i_plus)

  # now unwrap just the lstm till the second seq 
  while index < second_seq:
    # compress 
    mean_1, stddev_1, hidden = ring_net.lstm_compression(mean_1, stddev_1, hidden, 1.0)
    output_t_mean.append(mean_1)
    output_t_stddev.append(stddev_1)
    index = index + 1 
 
  # second unwrap 3 and get images 
  for i in xrange(3):
    # compress 
    mean_1, stddev_1, hidden = ring_net.lstm_compression(mean_1, stddev_1, hidden, 1.0)
    output_t_mean.append(mean_1)
    output_t_stddev.append(stddev_1)
    index = index + 1 
 
    # decode
    x_i_plus = ring_net.decoding(mean_1, stddev_1, 1.0)
    output_g.append(x_i_plus)

  # now unwrap just the lstm till the second seq 
  while index < third_seq:
    # compress 
    mean_1, stddev_1, hidden = ring_net.lstm_compression(mean_1, stddev_1, hidden, 1.0)
    output_t_mean.append(mean_1)
    output_t_stddev.append(stddev_1)
    index = index + 1 
  
  # second unwrap 3 and get images 
  for i in xrange(3):
    # compress 
    mean_1, stddev_1, hidden = ring_net.lstm_compression(mean_1, stddev_1, hidden, 1.0)
    output_t_mean.append(mean_1)
    output_t_stddev.append(stddev_1)
    index = index + 1 
 
    # decode
    x_i_plus = ring_net.decoding(mean_1, stddev_1, 1.0)
    output_g.append(x_i_plus)

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
  
  return output_f_mean, output_f_stddev, output_t_mean, output_t_stddev, output_g

def autoencoder(inputs, step=0):
  # encode and then decode
  y_0 = ring_net.encoding(inputs[:, step, :, :, :], 1.0) 
  x_1 = ring_net.decoding(y_0, 1.0)

  return x_1

