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

tf.app.flags.DEFINE_string('eval_dir', '../checkpoints/ring_net_eval_store',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('video_name', 'color_video.mov',
                           """name of the video you are saving""")

NUM_NETWORKS = 3
NETWORKS = ["../checkpoints/balls_2_paper_lstm_32x32x3balls_compressionkl_compression_vae_loss_False_sample_compression_False_lstm_size_64", "../checkpoints/balls_2_paper_lstm_32x32x3balls_compressionkl_compression_vae_loss_False_sample_compression_False_lstm_size_128", "../checkpoints/balls_2_paper_lstm_32x32x3balls_compressionkl_compression_vae_loss_False_sample_compression_False_lstm_size_256"]
MODELS = ["lstm_32x32x3", "lstm_32x32x3", "lstm_32x32x3"] 
LSTM_SIZE = [64, 128, 256]

NUM_FRAMES = 60
NUM_RUNS = 100

def evaluate(i):
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
    ckpt = tf.train.get_checkpoint_state(NETWORKS[i])
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("restored file from " + ckpt.model_checkpoint_path)
    else:
      print("no chekcpoint file found, this is an error")

    # start que runner
    tf.train.start_queue_runners(sess=sess)

    # error and stddev to keep track of
    error = np.zeros(NUM_FRAMES-1)
    error_blank = np.zeros(NUM_FRAMES-1)
    error_mean = np.zeros(NUM_FRAMES-1)
    stddev = np.zeros(NUM_FRAMES)

    # eval a few times
    for step in xrange(NUM_RUNS): 
      print("generated_seq")
      generated_seq, inputs, output_t_stddev_r, f_mean, t_mean = sess.run([output_g, x, output_t_stddev, output_f_mean, output_t_mean],feed_dict={})
      error = error + np.sum(np.square(generated_seq[:,:NUM_FRAMES-1,:,:,:] - inputs[:,1:,:,:,:]), axis=(0,2,3,4))
      error_blank = error_blank + np.sum(np.square(inputs[:,1:,:,:,:]), axis=(0,2,3,4))
      error_mean = error_mean + np.sum(np.square(f_mean[:,1:,:] - t_mean[:,:NUM_FRAMES-1,:]), axis=(0,2))
      stddev = stddev + np.sum(np.square(output_t_stddev_r), axis=(0,2))
      #generated_seq = generated_seq[0]
      #inputs = inputs[0]

    print(error)
    print(error.shape)
    error = error / NUM_RUNS
    error_blank = error_blank / NUM_RUNS
    stddev = stddev / NUM_RUNS
 
    plt.figure(1)
    plt.subplot(311)
    plt.plot(error[4:], label="lstm size " + str(LSTM_SIZE[i]))
    if i == 0:
      plt.plot(error_blank, label="blank image")
    plt.title("mean squared error")
    plt.legend()
    plt.subplot(312)
    plt.plot(stddev[4:], label="lstm size " + str(LSTM_SIZE[i]))
    plt.title("standard deviation")
    plt.legend()
    plt.subplot(313)
    plt.plot(error_mean[4:], label="lstm size " + str(LSTM_SIZE[i]))
    plt.title("mean squared error lstm")
    plt.legend()

    plt.savefig('../../Compressing-Dynamical-Systems-Paper/figures/bouncing_balls/bouncing_balls_error.png')

def main(argv=None):  # pylint: disable=unused-argument
  FLAGS.system = 'balls'
  FLAGS.num_balls = 2 
  FLAGS.train = False
  for i in xrange(NUM_NETWORKS):
    FLAGS.model = MODELS[i]
    FLAGS.lstm_size = LSTM_SIZE[i]
    evaluate(i)
  plt.show()


if __name__ == '__main__':
  tf.app.run()
