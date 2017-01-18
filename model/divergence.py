
import tensorflow as tf

def _simple_conv_2d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv2d(x, k, [1, 1, 1, 1], padding='VALID')
  return y

def spatial_divergence_2d(field):
  # implementation of spatial divergence
  # reimplemented from torch FluidNet implementation

  # make weight for x divergence
  weight_x_np = np.zeros([1,1,3,1])
  weight_x_np[0,0,0,0] = -1.0/2.0
  weight_x_np[0,0,1,0] = 0.0 
  weight_x_np[0,0,2,0] = 1.0/2.0

  weight_x = tf.constant(weight_x_np)

  # make weight for y divergence
  weight_y_np = np.zeros([1,3,1,1])
  weight_y_np[0,0,0,0] = -1.0/2.0
  weight_y_np[0,1,0,0] = 0.0 
  weight_y_np[0,2,0,0] = 1.0/2.0

  weight_y = tf.constant(weight_y_np)

  # calc gradientes
  field_dx = _simple_conv_2d(field, weight_x)
  field_dy = _simple_conv_2d(field, weight_y)

  # divergence of field
  field_div = field_dx + field_dy

  # kill boundrys (this is not correct! I should use boundrys but for right now I will not)
  field_div = field(:,1:-2,1:-2,:]

  return field_div

  






