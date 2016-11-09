
"""functions used to construct different losses
"""

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


def lstm_loss(inputs, output_f, output_t, output_g, output_autoencoding, train_piece):

  # stabilizing
  epsilon = 1e-10

  # autoencoder loss piece
  if train_piece == "autoencoder" or train_piece == "all":
    loss_reconstruction_autoencoder = 3000.0*tf.nn.l2_loss(inputs - output_autoencoding)
    #loss_reconstruction_autoencoder = tf.nn.l2_loss(inputs - output_autoencoding)
    tf.scalar_summary('loss_reconstruction_autoencoder', loss_reconstruction_autoencoder)
  else:
    loss_reconstruction_autoencoder = 0.0

  # compression loss piece
  seq_length = int(inputs.get_shape()[1])
  if (seq_length > 1) and (train_piece == "compression" or train_piece == "all"):
    if train_piece == "compression":
      output_f = tf.stop_gradient(output_f)
  
    #loss_t = 100.0*tf.nn.l2_loss(output_f[:,1:,:] - output_t[:,:seq_length-1,:])
    loss_t = tf.nn.l2_loss(output_f[:,5:,:] - output_t[:,4:seq_length-1,:])
    tf.scalar_summary('loss_t', loss_t)
  else:
    loss_t = 0.0

  total_loss = tf.reduce_sum(loss_reconstruction_autoencoder + loss_t)
  tf.scalar_summary('total_loss', total_loss)

  #return loss_vae, loss_reconstruction_autoencoder, loss_reconstruction_g, loss_t
  return total_loss 

