
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
tf.app.flags.DEFINE_string('model', 'fully_connected_28x28x4',
                           """ model name to train """)
tf.app.flags.DEFINE_bool('train', True,
                           """ model name to train """)
tf.app.flags.DEFINE_string('system', 'cannon',
                           """ system to compress """)
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          """The decay to use for the moving average""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """momentum of learning rate""")
tf.app.flags.DEFINE_float('alpha', 0.1,
                          """Leaky RElu param""")
tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)
tf.app.flags.DEFINE_float('beta', 0.1,
                          """ beta for loss value """)
tf.app.flags.DEFINE_integer('lstm_size', 10,
                          """ size of the lstm""")
tf.app.flags.DEFINE_integer('num_layers', 4,
                          """ size of the lstm""")
tf.app.flags.DEFINE_integer('compression_size', 10,
                          """ size of compressed space""")
tf.app.flags.DEFINE_bool('flow_open', False, 
                           """ whether flow is open """)

# train param
tf.app.flags.DEFINE_bool('test_no_sample', False, 
                           """ whether to sample during testing """)
tf.app.flags.DEFINE_bool('sample_compression', False, 
                           """ whether to sample compression """)
tf.app.flags.DEFINE_string('compression_loss', 'kl',
                           """ loss for the compression, either l2 or kl """)
tf.app.flags.DEFINE_bool('compression_vae_loss', False,
                           """ loss for the compression, either l2 or kl """)

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
  if FLAGS.system == "cannon":
    x = ring_net_input.cannon_inputs(batch_size, seq_length)
  elif FLAGS.system == "balls":
    x = ring_net_input.balls_inputs(batch_size, seq_length)
  elif FLAGS.system == "video":
    x = ring_net_input.video_inputs(batch_size, seq_length)
  elif FLAGS.system == "fluid":
    x = ring_net_input.fluid_inputs(batch_size, seq_length)
  elif FLAGS.system == "diffusion":
    x = ring_net_input.diffusion_inputs(batch_size, seq_length)
  return x

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
  return y_2, hidden 

def decoding(y_2):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_2 -> x_2
  if FLAGS.model in ("fully_connected_32x32x3", "lstm_32x32x3", "lstm_32x32x3_256", "lstm_32x32x3_512"): 
    x_2 = architecture.decoding_32x32x3(y_2)
  elif FLAGS.model in ("fully_connected_32x32x3", "lstm_32x32x1", "lstm_32x32x3_256", "lstm_32x32x3_512"): 
    x_2 = architecture.decoding_32x32x1(y_2)

  return x_2 

def encode_compress_decode(state, hidden_state, keep_prob_encoding, keep_prob_lstm):
  
  y_1 = encoding(state, keep_prob_encoding)
  y_2, hidden_state = lstm_compression(y_1, hidden_state, keep_prob_encoding)
  x_2 = decoding(y_2) 

  return x_2, hidden_state

def unwrap(inputs, keep_prob_encoding, keep_prob_lstm, keep_prob_decoding, seq_length, train_piece):
  """Unrap the system for training.
  Args:
    inputs: input to system, should be [minibatch, seq_length, image_size]
    keep_prob: dropout layers
    seq_length: how far to unravel 
 
  Return: 
    output_t: calculated y values from iterating t'
    output_g: calculated x values from g
    output_f: calculated y values from f 
  """

  if FLAGS.model in ("lstm_28x28x4", "lstm_84x84x4", "lstm_84x84x3", "lstm_32x32x3", "lstm_32x32x10", "lstm_64x64x10","lstm_32x32x3_256", "lstm_32x32x1"):
    output_f, output_t, output_g, output_autoencoder = unwrap_helper.lstm_unwrap(inputs, keep_prob_encoding, keep_prob_lstm, keep_prob_decoding, seq_length, train_piece)

  return output_f, output_t, output_g, output_autoencoder

def unwrap_generate(inputs, input_seq_length, output_seq_length):
  """Unrap the system for training.
  Args:
    inputs: input to system, should be [minibatch, seq_length, image_size]
    keep_prob: dropout layers
    seq_length: how far to unravel 
 
  Return: 
    output_t: calculated y values from iterating t'
    output_g: calculated x values from g
    output_f: calculated y values from f 
  """

  if FLAGS.model in ("lstm_28x28x4", "lstm_84x84x4", "lstm_84x84x3", "lstm_32x32x3", "lstm_32x32x10", "lstm_64x64x10", "lstm_32x32x3_256", "lstm_32x32x1"):
    output_f, output_t, output_g = unwrap_helper.lstm_unwrap_generate(inputs, input_seq_length, output_seq_length)

  return output_f, output_t, output_g

def unwrap_generate_3_skip(inputs, second_seq, third_seq):
  """Unrap the system for training.
  Args:
    inputs: input to system, should be [minibatch, seq_length, image_size]
    keep_prob: dropout layers
    seq_length: how far to unravel 
 
  Return: 
    output_t: calculated y values from iterating t'
    output_g: calculated x values from g
    output_f: calculated y values from f 
  """

  if FLAGS.model in ("lstm_32x32x3", "lstm_32x32x10", "lstm_64x64x10", "lstm_84x84x3", "lstm_84x84x4", "lstm_32x32x3_256", "lstm_32x32x1"): 
    output_f, output_t, output_g = unwrap_helper.lstm_unwrap_generate_3_skip(inputs, input_seq_length, output_seq_length)

  return output_f, output_t, output_g

def autoencoder(inputs, step=0):
  """Unrap the system for training.
  Args:
    inputs: input to system, should be [minibatch, seq_length, image_size]
    keep_prob: dropout layers
    seq_length: how far to unravel 
 
  Return: 
    output_t: calculated y values from iterating t'
    output_g: calculated x values from g
    output_f: calculated y values from f 
  """
  x_1 = unwrap_helper.autoencoder(inputs, step)

  return x_1 

def loss(inputs, output_f, output_t, output_g, output_autoencoder, train_piece):
  """Calc loss for unrap output.
  Args.
    inputs: true x values
    output_g: calculated x values from g
    output_mean: calculated mean values from f 
    output_stddev: calculated stddev values from f 

  Return:
    error: loss value
  """

  if FLAGS.model in ("lstm_84x84x4", "lstm_84x84x3", "lstm_32x32x3", "lstm_32x32x10", "lstm_64x64x10", "lstm_32x32x3_256", "lstm_32x32x1"):
    total_loss = loss_helper.lstm_loss(inputs, output_f, output_t, output_g, output_autoencoder, train_piece)
  
  #return loss_vae, loss_reconstruction_autoencoder, loss_reconstruction_g, loss_t
  return total_loss

def l2_loss(output, correct_output):
  """Calcs the loss for the model"""
  error = tf.nn.l2_loss(output - correct_output)
  return error
 
def train(total_loss, lr):
   #train_op = tf.train.AdamOptimizer(lr, epsilon=1.0).minimize(total_loss)
   optim = tf.train.AdamOptimizer(lr)
   train_op = optim.minimize(total_loss)
   return train_op

