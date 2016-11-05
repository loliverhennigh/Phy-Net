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

NUM_NETWORKS = 2
NETWORKS = ["../checkpoints/balls_1_l1_lstm_32x32x3balls_autoencoder", "../checkpoints/balls_2_l1_lstm_32x32x3balls_autoencoder"]
MODELS = ["lstm_32x32x3", "lstm_32x32x3"] 
LSTM_SIZE = [128, 128]

NUM_FRAMES = 1
NUM_RUNS = 1000

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

    # eval a few times
    
    for step in xrange(NUM_RUNS): 
      if step == 0:
        f_stddev_r = sess.run([output_f_stddev],feed_dict={})[0]
      else:
        f_stddev_r = f_stddev_r + sess.run([output_f_stddev],feed_dict={})[0]
     
    f_stddev_r = np.sort(f_stddev_r) / (NUM_RUNS*NUM_FRAMES)
 
    plt.figure(1)
    plt.plot(f_stddev_r[0,0,:])
    plt.show()

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
