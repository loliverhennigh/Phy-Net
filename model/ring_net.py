
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
tf.app.flags.DEFINE_string('model', 'lstm_32x32x3',
                           """ model name to train """)
tf.app.flags.DEFINE_bool('train', True,
                           """ train or test """)
tf.app.flags.DEFINE_string('system', 'diffusion',
                           """ system to compress """)


# possible models and systems to train are
# fully_connected_28x28x4 with cannon
# lstm_28x28x4 with cannon
# fully_connected_28x28x3 video with rgb
# lstm_28x28x3 video with rgb
# fully_connected_84x84x4 black and white video with 4 frames
# lstm_84x84x3 black and white video with 4 frames
# fully_connected_84x84x3 video with rgb
# lstm_84x84x3 video with rgb

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

