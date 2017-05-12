
"""functions used to construct different architectures  
"""


import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def int_shape(x):
  return list(map(int, x.get_shape()))

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat([x, -x], axis))

def set_nonlinearity(name):
  if name == 'concat_elu':
    return concat_elu
  elif name == 'elu':
    return tf.nn.elu
  elif name == 'concat_relu':
    return tf.nn.crelu
  elif name == 'relu':
    return tf.nn.relu
  else:
    raise('nonlinearity ' + name + ' is not supported')

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

def _variable(name, shape, initializer):
  """Helper to create a Variable.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  # getting rid of stddev for xavier ## testing this for faster convergence
  var = tf.get_variable(name, shape, initializer=initializer)
  return var

def conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = int(inputs.get_shape()[-1])
    
    # determine if 2d or 3d conv is needed
    length_input = len(inputs.get_shape())
    d2d = False
    d3d = False
    if length_input == 4:
      d2d = True
    elif length_input == 5:
      d3d = True
    else:
      print("conv layer does not support non 2d or 3d inputs")
      exit()

    if d2d: 
      weights = _variable('weights', shape=[kernel_size,kernel_size,input_channels,num_features],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    elif d3d:
      weights = _variable('weights', shape=[kernel_size,kernel_size,kernel_size,input_channels,num_features],initializer=tf.contrib.layers.xavier_initializer_conv2d())

    # pad it mobius
    if d2d:
      top = inputs[:,-1:]
      bottom = inputs[:,:1]
      inputs = tf.concat([top, inputs, bottom], axis=1)
      if FLAGS.system == "em":
        left = inputs[:,:,-1:] # make zero
        right = inputs[:,:,:1] # make zero
      else:
        left = tf.zeros_like(inputs[:,:,-1:]) # make zero
        right = tf.zeros_like(inputs[:,:,:1]) # make zero
      inputs = tf.concat([left, inputs, right], axis=2)

    if d3d:
      top = inputs[:,-1:]
      bottom = inputs[:,:1]
      inputs = tf.concat([top, inputs, bottom], axis=1)
      left = inputs[:,:,-1:]
      right = inputs[:,:,:1]
      inputs = tf.concat([left, inputs, right], axis=2)
      z_in = tf.zeros_like(inputs[:,:,:,-1:])
      z_out = tf.zeros_like(inputs[:,:,:,:1])
      inputs = tf.concat([z_in, inputs, z_out], axis=3)

    biases = _variable('biases',[num_features],initializer=tf.contrib.layers.xavier_initializer_conv2d())

    if d2d:
      conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='VALID')
    elif d3d:
      conv = tf.nn.conv3d(inputs, weights, strides=[1, stride, stride, stride, 1], padding='VALID')

    conv = tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
      conv = nonlinearity(conv)
    return conv

def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_channels = int(inputs.get_shape()[-1])
     
    # determine if 2d or 3d trans conv is needed
    length_input = len(inputs.get_shape())
    d2d = False
    d3d = False
    if length_input == 4:
      d2d = True
    elif length_input == 5:
      d3d = True
    else:
      print("transpose conv layer does not support non 2d or 3d inputs")
      exit()

    # pad it mobius
    if d2d:
      inputs_pad = inputs
      top = inputs_pad[:,-1:]
      bottom = inputs_pad[:,:1]
      inputs_pad = tf.concat([top, inputs_pad, bottom], axis=1)
      if FLAGS.system == "em":
        left = inputs_pad[:,:,-1:]
        right = inputs_pad[:,:,:1]
      else:
        left = tf.zeros_like(inputs_pad[:,:,-1:]) # make zero
        right = tf.zeros_like(inputs_pad[:,:,:1]) # make zero
      inputs_pad = tf.concat([left, inputs_pad, right], axis=2)

    if d3d:
      inputs_pad = inputs
      top = inputs[:,-1:]
      bottom = inputs[:,:1]
      inputs_pad = tf.concat([top, inputs_pad, bottom], axis=1)
      left = inputs_pad[:,:,-1:]
      right = inputs_pad[:,:,:1]
      inputs_pad = tf.concat([left, inputs_pad, right], axis=2)
      z_in = tf.zeros_like(inputs_pad[:,:,:,-1:])
      z_out = tf.zeros_like(inputs_pad[:,:,:,:1])
      inputs_pad = tf.concat([z_in, inputs_pad, z_out], axis=3)

    if d2d: 
      weights = _variable('weights', shape=[kernel_size,kernel_size,num_features,input_channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    elif d3d:
      weights = _variable('weights', shape=[kernel_size,kernel_size,kernel_size,num_features,input_channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())

    biases = _variable('biases',[num_features],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    batch_size = tf.shape(inputs)[0]

    if d2d:
      output_shape = tf.stack([tf.shape(inputs_pad)[0], tf.shape(inputs_pad)[1]*stride, tf.shape(inputs_pad)[2]*stride, num_features]) 
      conv = tf.nn.conv2d_transpose(inputs_pad, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
      conv = conv[:,2:-2,2:-2]
    elif d3d:
      output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs_pad)[1]*stride, tf.shape(inputs_pad)[2]*stride, tf.shape(inputs_pad)[3]*stride, num_features]) 
      conv = tf.nn.conv3d_transpose(inputs_pad, weights, output_shape, strides=[1,stride,stride,stride,1], padding='SAME')
      conv = conv[:,2:-2,2:-2,2:-2]

    conv_biased = tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
      conv_biased = nonlinearity(conv_biased)

    #reshape
    shape = int_shape(inputs)
    if d2d:
      conv_biased = tf.reshape(conv_biased, [shape[0], shape[1]*stride, shape[2]*stride, num_features])
    if d3d:
      conv_biased = tf.reshape(conv_biased, [shape[0], shape[1]*stride, shape[2]*stride, shape[3]*stride, num_features])

    return conv_biased

def fc_layer(inputs, hiddens, idx, nonlinearity=None, flat = False):
  with tf.variable_scope('{0}_fc'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs
    
    weights = _variable('weights', shape=[dim,hiddens],initializer=tf.contrib.layers.xavier_initializer())
    biases = _variable('biases', [hiddens], initializer=tf.contrib.layers.xavier_initializer())
    output_biased = tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')
    if nonlinearity is not None:
      output_biased = nonlinearity(ouput_biased)
    return output_biased

def nin(x, num_units, idx):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
    x = fc_layer(x, num_units, idx)
    return tf.reshape(x, s[:-1]+[num_units])

def _phase_shift(I, r):
  bsize, a, b, c = I.get_shape().as_list()
  bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
  X = tf.reshape(I, (bsize, a, b, r, r))
  X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
  X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
  X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
  X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
  X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
  return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, depth):
  Xc = tf.split(3, depth, X)
  X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
  return X

def trim_tensor(tensor, pos, width, trim_type):
  tensor_shape = int_shape(tensor)
  tensor_length = len(tensor_shape)
  if tensor_length == 4:
    if (pos-width < 0) or (pos+width+1 > max(tensor_shape[0],tensor_shape[1])):
      print("this should probably never be called")
      return tensor
    elif trim_type == "point":
      tensor = tensor[:,pos-width:pos+width+1,pos-width:pos+width+1]
    elif trim_type == "line":
      tensor = tensor[:,pos-width:pos+width+1]
    elif trim_type == "plane":
      print("can not extract a plane from a plane")
  elif tensor_length == 5:
    if (pos-width < 0) or (pos+width+1 > max(tensor_shape[0],tensor_shape[1],tensor_shape[2])):
      return tensor
    elif trim_type == "point":
      tensor = tensor[:,pos-width:pos+width+1,pos-width:pos+width+1,pos-width:pos+width+1]
    elif trim_type == "line":
      tensor = tensor[:,pos-width:pos+width+1,pos-width:pos+width+1]
    elif trim_type == "plane":
      tensor = tensor[:,pos-width:pos+width+1]
  else:
    print("tensor size not supported") 
    exit()
  return tensor

def res_block(x, a=None, filter_size=16, nonlinearity=concat_elu, keep_p=1.0, stride=1, gated=False, name="resnet", begin_nonlinearity=True):
      
  # determine if 2d or 3d trans conv is needed
  length_input = len(x.get_shape())
  d2d = False
  d3d = False
  if length_input == 4:
   d2d = True
  elif length_input == 5:
    d3d = True
  else:
    print("resnet does not support non 2d or 3d inputs")
    exit()

  orig_x = x
  if begin_nonlinearity: 
    x = nonlinearity(x) 
  if stride == 1:
    x = conv_layer(x, 3, stride, filter_size, name + '_conv_1')
  elif stride == 2:
    x = conv_layer(x, 4, stride, filter_size, name + '_conv_1')
  else:
    print("stride > 2 is not supported")
    exit()
  if a is not None:
    shape_a = int_shape(a) 
    shape_x_1 = int_shape(x)
    if d2d:
      a = tf.pad(
        a, [[0, 0], [0, shape_x_1[1]-shape_a[1]], [0, shape_x_1[2]-shape_a[2]],
        [0, 0]])
    elif d3d:
      a = tf.pad(
        a, [[0, 0], [0, shape_x_1[1]-shape_a[1]], [0, shape_x_1[2]-shape_a[2]], [0, shape_x_1[3]-shape_a[3]],
        [0, 0]])
    x += nin(nonlinearity(a), filter_size, name + '_nin')
  x = nonlinearity(x)
  if keep_p < 1.0:
    x = tf.nn.dropout(x, keep_prob=keep_p)
  if not gated:
    x = conv_layer(x, 3, 1, filter_size, name + '_conv_2')
  else:
    x = conv_layer(x, 3, 1, filter_size*2, name + '_conv_2')
    if d2d:
      x_1, x_2 = tf.split(x,2,3)
    elif d3d:
      x_1, x_2 = tf.split(x,2,4)
    x = x_1 * tf.nn.sigmoid(x_2)

  if int(orig_x.get_shape()[2]) > int(x.get_shape()[2]):
    assert(int(orig_x.get_shape()[2]) == 2*int(x.get_shape()[2]), "res net block only supports stirde 2")
    if d2d:
      orig_x = tf.nn.avg_pool(orig_x, [1,2,2,1], [1,2,2,1], padding='SAME')
    elif d3d:
      orig_x = tf.nn.avg_pool3d(orig_x, [1,2,2,2,1], [1,2,2,2,1], padding='SAME')

  # pad it
  out_filter = filter_size
  in_filter = int(orig_x.get_shape()[-1])
  if out_filter > in_filter:
    if d2d:
      orig_x = tf.pad(
          orig_x, [[0, 0], [0, 0], [0, 0],
          [(out_filter-in_filter), 0]])
    elif d3d:
      orig_x = tf.pad(
          orig_x, [[0, 0], [0, 0], [0, 0], [0, 0],
          [(out_filter-in_filter), 0]])
  elif out_filter < in_filter:
    orig_x = nin(orig_x, out_filter, name + '_nin_pad')

  return orig_x + x

def res_block_lstm(x, hidden_state_1=None, hidden_state_2=None, keep_p=1.0, name="resnet_lstm"):

  orig_x = x
  filter_size = orig_x.get_shape()

  with tf.variable_scope(name + "_conv_LSTM_1", initializer = tf.random_uniform_initializer(-0.01, 0.01)):
    lstm_cell_1 = BasicConvLSTMCell.BasicConvLSTMCell([int(x.get_shape()[1]),int(x.get_shape()[2])], [3,3], filter_size)
    if hidden_state_1 == None:
      batch_size = x.get_shape()[0]
      hidden_state_1 = lstm_cell_1.zero_state(batch_size, tf.float32) 

  x_1, hidden_state_1 = lstm_cell_1(x, hidden_state_1)
    
  if keep_p < 1.0:
    x_1 = tf.nn.dropout(x_1, keep_prob=keep_p)

  with tf.variable_scope(name + "_conv_LSTM_2", initializer = tf.random_uniform_initializer(-0.01, 0.01)):
    lstm_cell_2 = BasicConvLSTMCell.BasicConvLSTMCell([int(x_1.get_shape()[1]),int(x_1.get_shape()[2])], [3,3], filter_size)
    if hidden_state_2 == None:
      batch_size = x_1.get_shape()[0]
      hidden_state_2 = lstm_cell_2.zero_state(batch_size, tf.float32) 

  x_2, hidden_state_2 = lstm_cell_2(x_1, hidden_state_2)

  return orig_x + x_2, hidden_state_1, hidden_state_2

