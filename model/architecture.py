
"""functions used to construct different architectures  
"""

import tensorflow as tf
import numpy as np
import BasicConvLSTMCell

FLAGS = tf.app.flags.FLAGS

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable(name, shape, stddev):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  return var

def conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]

    weights = _variable('weights', shape=[kernel_size,kernel_size,input_channels,num_features],stddev=0.01)
    biases = _variable('biases',[num_features],stddev=0.01)

    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
      conv_biased = nonlinearity(conv_biased)
    return conv_biased

def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]
    
    weights = _variable('weights', shape=[kernel_size,kernel_size,num_features,input_channels], stddev=0.01)
    biases = _variable('biases',[num_features],stddev=0.01)
    batch_size = tf.shape(inputs)[0]
    output_shape = tf.pack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features]) 
    conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
      conv_biased = nonlinearity(conv_biased)
    return conv_biased

def fc_layer(inputs, hiddens, idx, flat = False, linear = False):
  with tf.variable_scope('{0}_fc'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs
    
    weights = _variable('weights', shape=[dim,hiddens],stddev=0.01)
    biases = _variable_on_cpu('biases', [hiddens], tf.constant_initializer(0.01))
    output_biased = tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')
    if nonlinearity is not None:
      ouput_biased = nonlinearity(ouput_biased)
    return ouput_biased

def _phase_shift(I, r):
  bsize, a, b, c = I.get_shape().as_list()
  bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
  X = tf.reshape(I, (bsize, a, b, r, r))
  X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
  X = tf.split(1, a, X)  # a, [bsize, b, r, r]
  X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
  X = tf.split(1, b, X)  # b, [bsize, a*r, r]
  X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
  return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, depth):
  Xc = tf.split(3, depth, X)
  X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
  return X

def int_shape(x):
  return list(map(int, x.get_shape()))

def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
    x = dense(x, num_units, **kwargs)
    return tf.reshape(x, s[:-1]+[num_units])

def res_block(x, a=None, filter_size=16, nonlinearity=concat_elu, keep_p=1.0, stride=1, gated=False, name="resnet"):
  orig_x = x
  x_1 = conv_layer(nonlinearity(x), 3, stride, filter_size, name + '_conv_1')
  if a is not None
    
    x_1 += nin(nonlinearity(a), filter_size)
  x_1 = nonlinearity(x_1)
  if keep_p < 1.0:
    x_1 = tf.nn.dropout(x_1, keep_prob=keep_p)
  if not gated:
    x_2 = conv_layer(x_1, 3, 1, filter_size, name + '_conv_2')
    return orig_x + x_2
  else:
    x_2 = conv_layer(x_1, 3, 1, filter_size*2, name + '_conv_2')
    x_2_1, x_2_2 = tf.split(3,2,x_2)
    x_2 = x_2_1 * tf.nn.sigmoid(x_2_2)
    return orig_x + x_2


def encoding_32x32x3(inputs, keep_prob):
  """Builds encoding part of ring net.
  Args:
    inputs: input to encoder
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  x_1_image = inputs 
  
  # conv1
  conv1 = _conv_layer(x_1_image, 8, 2, 64, "encode_1")
  # conv2
  conv2 = _conv_layer(conv1, 6, 2, 128, "encode_2")
  # conv3
  conv3 = _conv_layer(conv2, 6, 2, 128, "encode_3")
  # y_1 
  y_1 = _fc_layer(conv3, FLAGS.compression_size, "encode_4", True)
  _activation_summary(y_1)
  return y_1

def encoding_32x32x1(inputs, keep_prob):
  """Builds encoding part of ring net.
  Args:
    inputs: input to encoder
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  x_1_image = inputs 
  
  # conv1
  conv1 = _conv_layer(x_1_image, 3, 2, 8, "encode_1")
  # conv2
  conv2 = _conv_layer(conv1, 3, 1, 16, "encode_2")
  # y_1 
  y_1 = conv2
  _activation_summary(y_1)
  return y_1

def encoding_401x101x2(inputs, keep_prob):
  """Builds encoding part of ring net.
  Args:
    inputs: input to encoder
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  x_1_image = inputs 
  
  # conv1
  conv1 = _conv_layer(inputs, 3, 2, 64, "encode_1")
  # conv2
  conv2 = _conv_layer(conv1, 3, 1, 128, "encode_2")
  # conv3
  conv3 = _conv_layer(conv2, 3, 2, 256, "encode_3")
  # conv4
  conv4 = _conv_layer(conv3, 3, 1, 128, "encode_4")

  return conv4



def lstm_compression_32x32x3(y_1, hidden_state, keep_prob, encode=True):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  num_layers = FLAGS.num_layers

  y_1 = _fc_layer(y_1, FLAGS.lstm_size, "compression_1") 

  with tf.variable_scope("compress_LSTM", initializer = tf.random_uniform_initializer(-0.01, 0.01)):
    with tf.device('/gpu:0'):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.lstm_size, forget_bias=1.0)
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
      if hidden_state == None:
        batch_size = y_1.get_shape()[0]
        hidden_state = cell.zero_state(batch_size, tf.float32)

  y_2, new_state = cell(y_1, hidden_state)

  # residual connection
  #y_2 = y_2 + y_1

  y_2 = _fc_layer(y_2, FLAGS.compression_size, "compression_2") 

  return y_2, new_state

def lstm_compression_32x32x1(y_1, hidden_state, keep_prob, encode=True):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  y_1 = tf.nn.dropout(y_1, keep_prob)

  with tf.variable_scope("compress_LSTM", initializer = tf.random_uniform_initializer(-0.01, 0.01)):
    with tf.device('/gpu:0'):
      lstm_cell = BasicConvLSTMCell.BasicConvLSTMCell([int(y_1.get_shape()[1]),int(y_1.get_shape()[2])], [3,3], 16)
      if hidden_state == None:
        batch_size = y_1.get_shape()[0]
        hidden_state = lstm_cell.zero_state(batch_size, tf.float32) 

  y_2, new_state = lstm_cell(y_1, hidden_state)

  return y_2, new_state



def lstm_compression_401x101x2(y_1, hidden_state, keep_prob, encode=True):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  with tf.variable_scope("compress_LSTM", initializer = tf.random_uniform_initializer(-0.01, 0.01)):
    lstm_cell = BasicConvLSTMCell.BasicConvLSTMCell([int(y_1.get_shape()[1]),int(y_1.get_shape()[2])], [3,3], 128)
    if hidden_state == None:
      batch_size = y_1.get_shape()[0]
      hidden_state = lstm_cell.zero_state(batch_size, tf.float32) 

  y_2, new_state = lstm_cell(y_1, hidden_state)

  return y_2, new_state


def decoding_32x32x3(y_2):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_3 -> x_2

  # fc21
  fc21 = _fc_layer(y_2, 2048, "decode_21")
  conv21 = tf.reshape(fc21, [-1, 4, 4, 128])
  # conv22
  conv22 = _transpose_conv_layer(conv21, 6, 2, 128, "decode_22")
  # conv23
  conv23 = _transpose_conv_layer(conv22, 6, 2, 64, "decode_23")
  # conv24
  conv24 = _transpose_conv_layer(conv23, 6, 2, 3, "decode_24")
  x_2 = tf.reshape(conv24, [-1, 32, 32, 3])
  return x_2

def decoding_32x32x1(y_2):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_3 -> x_2

  # conv21
  conv21 = _transpose_conv_layer(y_2, 3, 1, 4, "decode_21")
  conv21 = tf.reshape(conv21, [-1, 16, 16, 4])
  conv21 = PS(conv21, 2, 1)
  #x_2 = x_2[:,:401,:101,:]
  x_2 = conv21
  return x_2

def decoding_gan_32x32x1(y_2, z):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_3 -> x_2

  # make z the right shape
  z_conv =  _fc_layer(z, 16*16, "decode_z")
  z_conv = tf.reshape(z_conv, [-1, 16, 16, 1])
  
  # concat z onto y_2
  y_2 = tf.concat(3, [y_2, z_conv])

  # one more layer and then do phase shift to get it out
  conv21 = _transpose_conv_layer(y_2, 3, 1, 4, "decode_21", True)
  conv21 = tf.reshape(conv21, [-1, 16, 16, 4])
  conv21 = PS(conv21, 2, 1)
  x_2 = conv21
  return x_2

def decoding_401x101x2(y_2):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_3 -> x_2

  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_3 -> x_2
  # conv21
  conv21 = _transpose_conv_layer(y_2, 3, 1, 128, "decode_21")
  # conv22
  conv22 = _transpose_conv_layer(conv21, 3, 1, 128, "decode_22")
  # conv23
  conv23 = _transpose_conv_layer(conv22, 3, 1, 64, "decode_23")
  # conv24
  conv24 = _transpose_conv_layer(conv23, 3, 1, 32, "decode_24", True)
  conv24 = tf.reshape(conv24, [-1, 101, 26, 32])
  conv24 = PS(conv24, 4, 2)
  #x_2 = x_2[:,:401,:101,:]
  x_2 = conv24[:,:401,:101,:]
  return x_2

def decoding_gan_401x101x2(y_2, z):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_3 -> x_2

  """

  # make z the right shape
  y_shape = y_2.get_shape()
  z_conv =  _fc_layer(z, int(y_shape[1])*int(y_shape[2]), "decode_z")
  z_conv = tf.reshape(z_conv, [-1, int(y_shape[1]), int(y_shape[2]), 1])

  # concat z onto y_2
  y_2 = tf.concat(3, [y_2, z_conv])

  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_3 -> x_2
  # conv21
  conv21 = _transpose_conv_layer(y_2, 3, 1, 128, "decode_21")
  # conv22
  conv22 = _transpose_conv_layer(conv21, 3, 1, 128, "decode_22")
  # conv23
  conv23 = _transpose_conv_layer(conv22, 3, 1, 64, "decode_23")
  # conv24
  conv24 = _transpose_conv_layer(conv23, 3, 1, 32, "decode_24", True)
  conv24 = tf.reshape(conv24, [-1, 101, 26, 32])
  conv24 = PS(conv24, 4, 2)
  #x_2 = x_2[:,:401,:101,:]
  x_2 = conv24[:,:401,:101,:]
  return x_2

# GAN Stuff
def discriminator_32x32x1(x, hidden_state, keep_prob):
  """Builds discriminator.
  Args:
    inputs: i
  """
  #--------- Making the net -----------
  # x_2 -> hidden_state

  # split x
  num_of_d = 8
  x_split = tf.split(0,num_of_d, x)
  label = []

  for i in xrange(num_of_d):
    # conv1
    conv1 = _conv_layer(x_split[i], 5, 2, 32, "discriminator_1_" + str(i))
    # conv2
    conv2 = _conv_layer(conv1, 5, 2, 64, "discriminator_2_" + str(i))
    
    y_1 = _fc_layer(conv2, 128, "discriminator_5_" + str(i), True)
    y_1 = tf.nn.dropout(y_1, keep_prob)
   
    with tf.variable_scope("discriminator_LSTM_" + str(i), initializer = tf.random_uniform_initializer(-0.01, 0.01)) as scope:
      #with tf.device('/gpu:0'):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(128, forget_bias=1.0)
      if hidden_state == None:
        batch_size = y_1.get_shape()[0]
        hidden_state = lstm_cell.zero_state(batch_size, tf.float32)
  
      y_2, new_state = lstm_cell(y_1, hidden_state)
  
    label.append(_fc_layer(y_2, 1, "discriminator_6_" + str(i), False, True))

  label = tf.pack(label)
  
  return label, new_state

def discriminator_401x101x2(x, hidden_state, keep_prob):
  """Builds discriminator.
  Args:
    inputs: i
  """
  #--------- Making the net -----------
  # x_2 -> hidden_state

  # split x
  num_of_d = 2
  x_split = tf.split(0,num_of_d, x)
  label = []

  for i in xrange(num_of_d):
    # conv1
    conv1 = _conv_layer(x, 5, 2, 64, "discriminator_1_" + str(i))
    # conv2
    conv2 = _conv_layer(conv1, 3, 2, 128, "discriminator_2_" + str(i))
    # conv3
    conv3 = _conv_layer(conv2, 3, 2, 256, "discriminator_3_" + str(i))
    # conv4
    conv4 = _conv_layer(conv3, 3, 2, 128, "discriminator_4_" + str(i))
  
    y_1 = _fc_layer(conv4, 256, "discriminator_5_" + str(i), True)
 
    with tf.variable_scope("discriminator_LSTM_" + str(i), initializer = tf.random_uniform_initializer(-0.01, 0.01)):
      #with tf.device('/gpu:0'):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, forget_bias=1.0)
      if hidden_state == None:
        batch_size = y_1.get_shape()[0]
        hidden_state = lstm_cell.zero_state(batch_size, tf.float32)

      y_2, new_state = lstm_cell(y_1, hidden_state)

    label.append(_fc_layer(y_2, 1, "discriminator_6_" + str(i), False, True))

  label = tf.pack(label)

  return label, new_state



