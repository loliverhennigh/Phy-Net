
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import architecture
import unwrap_helper
import loss_helper
import input.ring_net_input as ring_net_input

FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.

################# system params
tf.app.flags.DEFINE_string('system', 'diffusion',
                           """ system to compress """)
tf.app.flags.DEFINE_string('representation', 'lattice',
                           """ train on lattice state or possibly velocity or magnetic/electric fields """)
tf.app.flags.DEFINE_integer('lattice_size', 9,
                           """ size of lattice """)
tf.app.flags.DEFINE_string('dimension', '2d',
                           """ dimension of simulation (2d or 3d) """)

################# model params
## resnet params
tf.app.flags.DEFINE_bool('residual', False,
                           """ residual connections """)
tf.app.flags.DEFINE_bool('residual_lstm', False,
                           """ residual connections around lstm """)
tf.app.flags.DEFINE_integer('nr_residual', 1,
                           """ number of residual blocks before down sizing """)
tf.app.flags.DEFINE_integer('nr_downsamples', 3,
                           """ numper of downsamples """)
tf.app.flags.DEFINE_bool('multi_resolution', False,
                           """ skip connections over resolutions """)
tf.app.flags.DEFINE_sting('nonlinearity', "concat_elu",
                           """ what nonlinearity to use, leakey_relu, relu, elu, concat_elu """)
## gan params
tf.app.flags.DEFINE_bool('gan', False,
                           """ use gan training """)
tf.app.flags.DEFINE_integer('nr_discriminators', 1,
                           """ number of discriminators to train """)
tf.app.flags.DEFINE_integer('z_size', 50,
                           """ size of z vector """)
## compression train
tf.app.flags.DEFINE_bool('compression', False,
                           """ train in compression style """)
## lstm params
tf.app.flags.DEFINE_bool('lstm', True,
                           """ lstm or just fully connected""")
tf.app.flags.DEFINE_integer('nr_lstm_layer', 1,
                           """ number of lstm layers """)

################# optimize params
tf.app.flags.DEFINE_sting('optimizer', "adam",
                           """ what optimizer to use """)
tf.app.flags.DEFINE_float('learning_rate_reconstruction', 1e-5,
                           """ learning rete for reconstruction """)
tf.app.flags.DEFINE_float('learning_rate_gan', 2e-5,
                           """ learning rate for training gan """)
tf.app.flags.DEFINE_float('lambda_reconstruction', 1.0,
                           """ weight of reconstruction error """) 
tf.app.flags.DEFINE_float('lambda_divergence', 0.2,
                           """ weight of divergence error """)

################# train params
tf.app.flags.DEFINE_integer('init_unroll_length', 5,
                           """ unroll length to initialize network """)
tf.app.flags.DEFINE_integer('unroll_length', 5,
                           """ unroll length """)
tf.app.flags.DEFINE_bool('unroll_true', True,
                           """ use the true data when unrolling the network (probably just used for unroll_length 1 when doing curriculum learning""")
tf.app.flags.DEFINE_integer('restore_unroll_length', 0,
                           """ what to unroll length to restore from. (if 0 then initialize from scratch) """)
tf.app.flags.DEFINE_integer('batch_size', 4,
                           """ batch size """)

################# input params
tf.app.flags.DEFINE_bool('train', True,
                           """ train or test """)

def inputs(batch_size, seq_length):
  """makes input vector
  Return:
    x: input vector, may be filled 
  """
  if FLAGS.system == "balls":
    return ring_net_input.balls_inputs(batch_size, seq_length)
  elif FLAGS.system == "diffusion":
    return ring_net_input.diffusion_inputs(batch_size, seq_length)
  elif FLAGS.system == "fluid":
    return ring_net_input.fluid_inputs(batch_size, seq_length)

def encoding(inputs, keep_prob_encoding):
  """Builds encoding part of ring net.
  Args:
    inputs: input to encoder
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  if FLAGS.model == "lstm_32x32x3":
    y_1 = architecture.encoding_32x32x3(inputs, keep_prob_encoding)
  elif FLAGS.model == "lstm_32x32x1":
    y_1 = architecture.encoding_32x32x1(inputs, keep_prob_encoding)
  elif FLAGS.model == "lstm_401x101x2":
    y_1 = architecture.encoding_401x101x2(inputs, keep_prob_encoding)

  return y_1 

def lstm_compression(y_1, hidden_state, keep_prob_lstm, encode=True):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  if FLAGS.model == "lstm_32x32x3":
    y_2, hidden = architecture.lstm_compression_32x32x3(y_1, hidden_state, keep_prob_lstm, encode)
  elif FLAGS.model == "lstm_32x32x1":
    y_2, hidden = architecture.lstm_compression_32x32x1(y_1, hidden_state, keep_prob_lstm, encode)
  elif FLAGS.model == "lstm_401x101x2":
    y_2, hidden = architecture.lstm_compression_401x101x2(y_1, hidden_state, keep_prob_lstm, encode)
  return y_2, hidden 

def decoding(y_2):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_2 -> x_2
  if FLAGS.model in ("lstm_32x32x3"): 
    x_2 = architecture.decoding_32x32x3(y_2)
  elif FLAGS.model in ("lstm_32x32x1"): 
    x_2 = architecture.decoding_32x32x1(y_2)
  elif FLAGS.model in ("lstm_401x101x2"): 
    x_2 = architecture.decoding_401x101x2(y_2)

  return x_2 

def decoding_gan(y_2, z):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_2 -> x_2
  if FLAGS.model in ("lstm_32x32x3"): 
    x_2 = architecture.decoding_gan_32x32x3(y_2, z)
  elif FLAGS.model in ("lstm_32x32x1"): 
    x_2 = architecture.decoding_gan_32x32x1(y_2, z)
  elif FLAGS.model in ("lstm_401x101x2"): 
    x_2 = architecture.decoding_gan_401x101x2(y_2, z)

  return x_2 

def encode_compress_decode(state, hidden_state, keep_prob_encoding, keep_prob_lstm):
  
  y_1 = encoding(state, keep_prob_encoding)
  y_2, hidden_state = lstm_compression(y_1, hidden_state, keep_prob_lstm)
  x_2 = decoding(y_2) 

  return x_2, hidden_state

def encode_compress_decode_gan(state, hidden_state, z, keep_prob_encoding, keep_prob_lstm):
  
  y_1 = encoding(state, keep_prob_encoding)
  y_2, hidden_state = lstm_compression(y_1, hidden_state, keep_prob_lstm)
  x_2 = decoding_gan(y_2, z) 

  return x_2, hidden_state

def discriminator(output, hidden_state, keep_prob_discriminator):

  if FLAGS.model in ("lstm_32x32x3"):
    label, hidden_state = architecture.discriminator_32x32x3(output, hidden_state, keep_prob_discriminator)
  elif FLAGS.model in ("lstm_32x32x1"):
    label, hidden_state = architecture.discriminator_32x32x1(output, hidden_state, keep_prob_discriminator)
  elif FLAGS.model in ("lstm_401x101x2"):
    label, hidden_state = architecture.discriminator_401x101x2(output, hidden_state, keep_prob_discriminator)
  return label, hidden_state 
  

def train(total_loss, lr):
   #train_op = tf.train.AdamOptimizer(lr, epsilon=1.0).minimize(total_loss)
   optim = tf.train.AdamOptimizer(lr)
   train_op = optim.minimize(total_loss)
   return train_op

