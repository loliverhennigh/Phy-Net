import math

import numpy as np
import tensorflow as tf
import cv2


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

NUM_NETWORKS = 1
#NETWORKS = ("../checkpoints/goldfish_paper_fully_connected_84x84x3video", "../checkpoints/goldfish_paper_lstm_84x84x3video") 
#MODELS = ("fully_connected_84x84x3", "lstm_84x84x3") 
#NETWORKS = ("../checkpoints/balls_paper_fully_connected_32x32x3balls", "../checkpoints/balls_paper_lstm_32x32x3balls") 
NETWORKS = ("../checkpoints/balls_1_paper_lstm_32x32x3balls_all_l2_compression_vae_loss_False_sample_compression_False", "../checkpoints/balls_1_paper_lstm_32x32x3balls_all_kl_compression_vae_loss_False_sample_compression_False", "../checkpoints/balls_1_paper_lstm_32x32x3balls_all_l2_compression_vae_loss_True_sample_compression_False", "../checkpoints/balls_1_paper_lstm_32x32x3balls_all_kl_compression_vae_loss_True_sample_compression_False") 
#MODELS = ("fully_connected_32x32x3", "lstm_32x32x3") 
MODELS = ("lstm_32x32x3", "lstm_32x32x3", "lstm_32x32x3", "lstm_32x32x3") 
#NETWORKS = ("../checkpoints/balls_paper_fully_connected_32x32x3balls", "../checkpoints/balls_paper_lstm_32x32x3balls") 
#MODELS = ("fully_connected_32x32x3", "lstm_32x32x3") 

assert(FLAGS.model in ("fully_connected_84x84x4", "fully_connected_84x84x3", "lstm_84x84x4", "lstm_84x84x3"), "need to use a model thats 84x84, sorry")

NUM_FRAMES = 100 
NUM_RUNS = 100

def evaluate(i):
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    x = ring_net.inputs(1, NUM_FRAMES) 
    # unwrap it
    output_f_mean, output_f_stddev, output_t_mean, output_t_stddev, output_g = ring_net.unwrap_generate(x, 2, NUM_FRAMES-2) 

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

    max_stddev = 0.0
    # eval a few times
    for step in xrange(NUM_RUNS): 
      print("generated_seq")
      generated_seq, inputs, output_t_stddev_r = sess.run([output_g, x, output_t_stddev],feed_dict={})
      stddev = np.sum(np.square(output_t_stddev_r), axis=(0,2))
      index = np.argmax(stddev)
      if max_stddev < stddev[index]:
        max_stddev = stddev[index] 
        highest_stddev_im = generated_seq[0,index,:,:,:]
        print(highest_stddev_im.shape)

    highest_stddev_im = np.uint8(np.abs(highest_stddev_im) * 255)
    print(np.max(highest_stddev_im))
    cv2.imwrite("high_stddev_image_" + MODELS[i] + ".jpg", highest_stddev_im)
      
      #generated_seq = generated_seq[0]
      #inputs = inputs[0]
 
def main(argv=None):  # pylint: disable=unused-argument
  for i in xrange(NUM_NETWORKS):
    FLAGS.model = MODELS[i]
    evaluate(i)


if __name__ == '__main__':
  tf.app.run()
