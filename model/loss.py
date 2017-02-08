
"""functions used to construct different losses
"""

import tensorflow as tf
import numpy as np
from divergence import spatial_divergence_2d


FLAGS = tf.app.flags.FLAGS


def loss_mse(true, generated):
  loss = tf.nn.l2_loss(true - generated)
  tf.summary.scalar('reconstruction_loss', loss)
  return loss
 
def loss_divergence(field):
  if len(field.get_shape()) == 5:
    field_div = spatial_divergence_2d(field)
  loss = tf.nn.l2_loss(field_div)
  tf.summary.scalar('divergence_loss', loss)
  return loss

def loss_gradient_difference(true, generated):
  # seen in Beyond Mean Square Error
  if len(true.get_shape()) == 5:
    true_x_shifted_right = true[:,:,1:,:,:]
    true_x_shifted_left = true[:,:,:-1,:,:]
    true_x_gradient = tf.abs(true_x_shifted_right - true_x_shifted_left)

    generated_x_shifted_right = generated[:,:,1:,:,:]
    generated_x_shifted_left = generated[:,:,:-1,:,:]
    generated_x_gradient = tf.abs(generated_x_shifted_right - generated_x_shifted_left)

    loss_x_gradient = tf.nn.l2_loss(true_x_gradient - generated_x_gradient)
    tf.summary.scalar('loss_x_gradient', loss_x_gradient)

    true_y_shifted_right = true[:,:,:,1:,:]
    true_y_shifted_left = true[:,:,:,:-1,:]
    true_y_gradient = tf.abs(true_y_shifted_right - true_y_shifted_left)

    generated_y_shifted_right = generated[:,:,:,1:,:]
    generated_y_shifted_left = generated[:,:,:,:-1,:]
    generated_y_gradient = tf.abs(generated_y_shifted_right - generated_y_shifted_left)
    
    loss_y_gradient = tf.nn.l2_loss(true_y_gradient - generated_y_gradient)
    tf.summary.scalar('loss_y_gradient', loss_y_gradient)

    loss = loss_x_gradient + loss_y_gradient
    tf.summary.scalar('gradient_loss', loss)

  return loss

  
  
def loss_gan_true(true_label, generated_label):
  loss_d_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(true_label, tf.ones_like(true_label)))
  loss_d_generated = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(generated_label, tf.ones_like(generated_label)))
  loss_d = loss_d_true + loss_d_generated
  tf.summary.scalar('error discriminator true', loss_d_true)
  tf.summary.scalar('error discriminator generated', loss_d_generated)
  tf.summary.scalar('error discriminator', loss_d)
  return loss_d
 
def loss_gan_generated(generated_label):
  error_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(generated_label, tf.ones_like(generated_label))) 
  tf.summary.scalar('error generated', error_g)
  return loss_g



