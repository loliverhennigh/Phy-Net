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

NUM_NETWORKS = 3
NETWORKS = ["../checkpoints/train_store_lstm_32x32x3balls_all_kl_compression_vae_loss_False_sample_compression_False_lstm_size_32_num_layers_4"]
MODELS = ["lstm_32x32x3"] 
LSTM_SIZE = [32]
NUM_LAYERS = [4]

NUM_FRAMES = 60
NUM_RUNS = 100

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    x = ring_net.inputs(1, NUM_FRAMES) 

    # unwrap it
    output_f, output_t, output_g, output_autoencoding = ring_net.unwrap(x, 1.0, 1.0, 1.0, NUM_FRAMES, "all") 

    # restore network
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(NETWORKS[0])
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
      plt.savefig('ball_pics_' + str(i) + '.png')
       
      #plt.show()
  
def main(argv=None):  # pylint: disable=unused-argument
  FLAGS.system = 'balls'
  FLAGS.num_balls = 2 
  FLAGS.train = False
  FLAGS.model = MODELS[0]
  FLAGS.lstm_size = LSTM_SIZE[0]
  FLAGS.num_layers = NUM_LAYERS[0]
  evaluate()


if __name__ == '__main__':
  tf.app.run()
