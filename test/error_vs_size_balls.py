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


if FLAGS.system='balls'
  NUM_NETWORKS = 3
  NETWORKS = ["../checkpoints/balls_2_paper_lstm_32x32x3balls_compressionkl_compression_vae_loss_False_sample_compression_False_lstm_size_64", "../checkpoints/balls_2_paper_lstm_32x32x3balls_compressionkl_compression_vae_loss_False_sample_compression_False_lstm_size_128", "../checkpoints/balls_2_paper_lstm_32x32x3balls_compressionkl_compression_vae_loss_False_sample_compression_False_lstm_size_256"]
  MODELS = ["lstm_32x32x3", "lstm_32x32x3", "lstm_32x32x3"]
  LSTM_SIZE = [64, 128, 256] 
  NUM_LAYERS = [3, 3, 3] 

#assert(FLAGS.model in ("fully_connected_84x84x4", "fully_connected_84x84x3", "lstm_84x84x4", "lstm_84x84x3"), "need to use a model thats 84x84, sorry")

NUM_FRAMES = 15
NUM_RUNS = 100

def evaluate(i):
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    x = ring_net.inputs(1, NUM_FRAMES) 
    # unwrap it
    output_f, output_t, output_g, output_autoencoder = ring_net.unwrap(x, 1.0, 1.0, 1.0, NUM_FRAMES, "all")

    # restore network
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(NETWORKS[i])
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("restored file from " + ckpt.model_checkpoint_path)
    else:
      print(ckpt.model_checkpoint_path)
      print(NETWORKS[i])
      print("no chekcpoint file found, this is an error")

    # start que runner
    tf.train.start_queue_runners(sess=sess)

    # error and stddev to keep track of
    error = np.zeros(NUM_FRAMES-6)

    # eval a few times
    for step in xrange(NUM_RUNS):
      print("generated_seq")
      output_g_g, x_g = sess.run([output_g, x],feed_dict={})
      error = error + np.sum(np.square(output_g_g[:,5:NUM_FRAMES-1,:,:,:] - x_g[:,6:,:,:,:]), axis=(0,2,3,4))
      #stddev = stddev + np.sum(np.square(output_t_stddev_r), axis=(0,2))
      #generated_seq = generated_seq[0]
      #inputs = inputs[0]
 
    plt.figure(1)
    plt.plot(error, label="error_" + NETWORKS[i])
    '''if FLAGS.system == "balls":
      plt.plot(error_blank, label="error_blank_" + NETWORKS[i])
    plt.legend()
    plt.figure(2)
    plt.plot(stddev, label="stddev_" + NETWORKS[i])
    plt.legend()
    plt.figure(3)
    plt.plot(error_mean, label="mean_" + NETWORKS[i])
    plt.legend()'''
    return np.sum(error)

def main(argv=None):  # pylint: disable=unused-argument
  computation = np.zeros(NUM_NETWORKS)
  error = np.zeros(NUM_NETWORKS)
  for i in xrange(NUM_NETWORKS):
    FLAGS.model = MODELS[i]
    FLAGS.lstm_size = LSTM_SIZE[i]
    FLAGS.num_layers = NUM_LAYERS[i]
    computation[i] = 3*LSTM_SIZE[i]^2*NUM_LAYERS[i] # just a estimation. possible more details later
    error[i] = evaluate(i)
  plt.figure(2)
  plt.scatter(computation, error)
  plt.show()


if __name__ == '__main__':
  tf.app.run()
