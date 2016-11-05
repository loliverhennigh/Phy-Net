import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 


import sys
sys.path.append('../')
import systems.cannon as cn
import systems.video as vi 

import model.ring_net as ring_net

FLAGS = tf.app.flags.FLAGS

NUM_NETWORKS = 1
NETWORKS = "../checkpoints/diffusion_lstm_32x32x1diffusion_compressionkl_compression_vae_loss_False_sample_compression_False_lstm_size_100"
#NETWORKS = "../checkpoints/diffusion_lstm_32x32x1diffusion_all_kl_compression_vae_loss_False_sample_compression_False_lstm_size_100"
MODELS = "lstm_32x32x1"
LSTM_SIZE = 100

NUM_FRAMES = 60

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    x = ring_net.inputs(1, NUM_FRAMES) 

    # unwrap it
    output_f_mean, output_f_stddev, output_t_mean, output_t_stddev, output_g, output_autoencoding = ring_net.unwrap(x, 1.0, 1.0, 1.0, NUM_FRAMES, "all") 

    # restore network
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(NETWORKS)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("restored file from " + ckpt.model_checkpoint_path)
    else:
      print("no chekcpoint file found, this is an error")

    # start que runner
    tf.train.start_queue_runners(sess=sess)

    # eval ounce
    generated_inputs, inputs, autoencoding = sess.run([output_g, x, output_autoencoding],feed_dict={})
    generated_inputs = generated_inputs[0]
    inputs = inputs[0]
    autoencoding = autoencoding[0]
    print(inputs.shape)

    # create grid
    y_p, x_p = np.mgrid[0:1:32j, 0:1:32j]

    # now plot say 5 25 and 45
    for i in (5, 25, 45):
      plt.figure(1)
      plt.subplot(131)
      plt.pcolor(x_p, y_p, inputs[i+1, :, :, 0], cmap='RdBu')
      plt.title("True")
      plt.colorbar()
      plt.subplot(132)
      plt.pcolor(x_p, y_p, generated_inputs[i, :, :, 0], cmap='RdBu')
      plt.title("generated")
      plt.colorbar()
      plt.subplot(133)
      plt.pcolor(x_p, y_p, np.square(generated_inputs[i, :, :, 0] - inputs[i+1, :, :, 0]), cmap='RdBu')
      plt.title("error")
      plt.colorbar()
      #plt.savefig('../../Compressing-Dynamical-Systems-Paper/figures/diffusion/diffusion_pics_' + str(i) + '.png')
       
  
def main(argv=None):  # pylint: disable=unused-argument
  FLAGS.system = 'diffusion'
  FLAGS.train = False
  FLAGS.model = MODELS
  FLAGS.lstm_size = LSTM_SIZE
  evaluate()
  plt.show()


if __name__ == '__main__':
  tf.app.run()
