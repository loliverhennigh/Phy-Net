import math

import numpy as np
import tensorflow as tf
import cv2


import sys
sys.path.append('../')
import systems.cannon as cn
import systems.video as vi 

import model.ring_net as ring_net
import model.unwrap_helper_test as unwrap_helper_test 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../checkpoints/ring_net_eval_store',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/train_store_',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('video_name', 'color_video.mov',
                           """name of the video you are saving""")


fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

success = video.open(FLAGS.video_name, fourcc, 4, (84, 84), True)

NUM_FRAMES = 500

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    x = ring_net.inputs(1, 1)
    # unwrap it
    keep_prob = tf.placeholder("float")
    y_0 = unwrap_helper_test.encoding(x, keep_prob)
    if FLAGS.model in ("lstm_28x28x4", "lstm_84x84x4", "lstm_84x84x3"):
      x_1, y_1, hidden_state_1 = unwrap_helper_test.lstm_step(y_0, None,  keep_prob)
    elif FLAGS.model in ("fully_connected_28x28x4", "fully_connected_84x84x4", "fully_connected_84x84x3"):
      x_1, y_1 = unwrap_helper_test.fully_connected_step(y_0,  keep_prob)
    # set reuse to true 
    tf.get_variable_scope().reuse_variables()
    if FLAGS.model in ("lstm_28x28x4", "lstm_84x84x4", "lstm_84x84x3"):
      x_2, y_2, hidden_state_2 = unwrap_helper_test.lstm_step(y_1, hidden_state_1,  keep_prob)
    elif FLAGS.model in ("fully_connected_28x28x4", "fully_connected_84x84x4", "fully_connected_84x84x3"):
      x_2, y_2 = unwrap_helper_test.fully_connected_step(y_1,  keep_prob)

    # restore network
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir + FLAGS.model + FLAGS.system)
    #ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("restored file from " + ckpt.model_checkpoint_path)
    else:
      print("no chekcpoint file found, this is an error")

    # start que runner
    tf.train.start_queue_runners(sess=sess)

    # eval ounce
    if FLAGS.model in ("lstm_28x28x4", "lstm_84x84x4", "lstm_84x84x3"):
      generated_x_1, generated_y_1, generated_hidden_state_1 = sess.run([x_1, y_1, hidden_state_1],feed_dict={keep_prob:1.0})
    elif FLAGS.model in ("fully_connected_28x28x4", "fully_connected_84x84x4", "fully_connected_84x84x3"):
      generated_x_1, generated_y_1 = sess.run([x_1, y_1],feed_dict={keep_prob:1.0})
    new_im = np.uint8(np.abs(generated_x_1/np.amax(generated_x_1[0, :, :, :]) * 255))
    video.write(new_im[0,:,:,:])
 
    # make video
    for step in xrange(NUM_FRAMES-1):
      # continue to calc frames
      print(step)
      if FLAGS.model in ("lstm_28x28x4", "lstm_84x84x4", "lstm_84x84x3"):
        generated_x_1, generated_y_1, generated_hidden_state_1 = sess.run([x_2, y_2, hidden_state_2],feed_dict={keep_prob:1.0, y_1:generated_y_1, hidden_state_1:generated_hidden_state_1})
      elif FLAGS.model in ("fully_connected_28x28x4", "fully_connected_84x84x4", "fully_connected_84x84x3"):
        generated_x_1, generated_y_1 = sess.run([x_2, y_2],feed_dict={keep_prob:1.0, y_1:generated_y_1})
      new_im = np.uint8(np.abs(generated_x_1/np.amax(generated_x_1[0, :, :, :]) * 255))
      video.write(new_im[0,:,:,0:3])
    print('saved to ' + FLAGS.video_name)
    video.release()
    cv2.destroyAllWindows()
       
def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
