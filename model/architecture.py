
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

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/gpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    weight_decay.set_shape([])
    tf.add_to_collection('losses', weight_decay)
  return var

def _conv_layer(inputs, kernel_size, stride, num_features, idx, linear = False):
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]

    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,input_channels,num_features],stddev=0.01, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.01))

    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    if linear:
      return conv_biased
    conv_rect = tf.nn.elu(conv_biased,name='{0}_conv'.format(idx))
    return conv_rect

def _transpose_conv_layer(inputs, kernel_size, stride, num_features, idx, linear = False):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]
    
    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,num_features,input_channels], stddev=0.01, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.01))
    batch_size = tf.shape(inputs)[0]
    output_shape = tf.pack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features]) 
    conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    if linear:
      return conv_biased
    conv_rect = tf.nn.elu(conv_biased,name='{0}_transpose_conv'.format(idx))
    return conv_rect
     

def _fc_layer(inputs, hiddens, idx, flat = False, linear = False):
  with tf.variable_scope('{0}_fc'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs
    
    weights = _variable_with_weight_decay('weights', shape=[dim,hiddens],stddev=0.01, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases', [hiddens], tf.constant_initializer(0.01))
    if linear:
      return tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')
  
    ip = tf.add(tf.matmul(inputs_processed,weights),biases)
    return tf.nn.elu(ip,name=str(idx)+'_fc')

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
    with tf.device('/gpu:0'):
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

def decoding_401x101x2(y_2):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_3 -> x_2

  # fc21
  fc21 = _fc_layer(y_2, 17*5*128, "decode_21")
  conv21 = tf.reshape(fc21, [-1, 17, 5, 128])
  # conv22
  conv22 = _transpose_conv_layer(conv21, 6, 2, 128, "decode_22")
  # conv23
  conv23 = _transpose_conv_layer(conv22, 6, 3, 64, "decode_23")
  # conv24
  conv24 = _transpose_conv_layer(conv23, 8, 4, 2, "decode_24")
  x_2 = tf.reshape(conv24, [-1, 17*4*3*2, 5*4*3*2, 2])
  x_2 = x_2[:,:401,:101,:]
  return x_2
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

# GAN Stuff
def discriminator_lstm(x, hidden_state):
  """Builds discriminator.
  Args:
    inputs: i
  """
  #--------- Making the net -----------
  # x_2 -> hidden_state
  # conv1
  conv1 = _conv_layer(x, 3, 2, 64, "discriminator_1")
  # conv2
  conv2 = _conv_layer(conv1, 3, 2, 128, "discriminator_2")
  # conv3
  conv3 = _conv_layer(conv2, 3, 2, 256, "discriminator_3")
  # conv4
  conv4 = _conv_layer(conv3, 3, 2, 128, "discriminator_4")
  
  y_1 = _fc_layer(conv4, 256, "discriminator_5", True)
 
  with tf.variable_scope("compress_LSTM", initializer = tf.random_uniform_initializer(-0.01, 0.01)):
    with tf.device('/gpu:0'):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, forget_bias=1.0)
      if hidden_state == None:
        batch_size = y_1.get_shape()[0]
        hidden_state = lstm_cell.zero_state(batch_size, tf.float32)

  y_2, new_state = lstm_cell(y_1, hidden_state)

  label = _fc_layer(y_2, 2, "discriminator_6")

  return label, new_state



