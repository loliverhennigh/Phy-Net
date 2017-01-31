
import tensorflow as tf
import numpy as np

from nn import int_shape

def _simple_conv_2d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y

def spatial_divergence_2d(field):
  # implementation of spatial divergence
  # reimplemented from torch FluidNet implementation
  field_shape = int_shape(field)
  field = tf.reshape(field, [field_shape[0]*field_shape[1], field_shape[2], field_shape[3], field_shape[4]])

  # make weight for x divergence
  weight_x_np = np.zeros([3,1,3,1])
  weight_x_np[0,0,0,0] = -1.0/2.0
  weight_x_np[1,0,0,0] = 0.0 
  weight_x_np[2,0,0,0] = 1.0/2.0

  weight_x = tf.constant(np.float32(weight_x_np))

  # make weight for y divergence
  weight_y_np = np.zeros([1,3,3,1])
  weight_y_np[0,0,1,0] = -1.0/2.0
  weight_y_np[0,1,1,0] = 0.0 
  weight_y_np[0,2,1,0] = 1.0/2.0

  weight_y = tf.constant(np.float32(weight_y_np))

  # calc gradientes
  print(field.get_shape())
  print(weight_y.get_shape())
  field_dx = _simple_conv_2d(field, weight_x)
  field_dy = _simple_conv_2d(field, weight_y)

  # divergence of field
  field_div = field_dx + field_dy

  # kill boundrys (this is not correct! I should use boundrys but for right now I will not)
  field_div = tf.abs(field_div[:,1:-2,1:-2,:])

  return field_div

