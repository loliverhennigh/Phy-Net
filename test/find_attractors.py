import math

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

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
tf.app.flags.DEFINE_integer('run_length', 300,
                           """number of frames to run out""")
tf.app.flags.DEFINE_integer('num_runs', 1000,
                           """number of frames to run out""")

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()
error_t = []
error_f =[]
error_tf =[]

if FLAGS.model in ("fully_connectd_28x28x4", "lstm_28x28x4"):
  success = video.open(FLAGS.video_name, fourcc, 1, (28, 28), True)
  shape = (28,28)
  frame_num = 4
  color = False
elif FLAGS.model in ("fully_connected_84x84x4", "lstm_84x84x4"):
  #success = video.open(FLAGS.video_name, fourcc, 4, (252, 84), True)
  success = video.open(FLAGS.video_name, fourcc, 1, (84,84), True)
  shape = (84,84)
  frame_num = 4
  color = False
elif FLAGS.model in ("fully_connected_84x84x3", "lstm_84x84x3"):
  #success = video.open(FLAGS.video_name, fourcc, 4, (252, 84), True)
  success = video.open(FLAGS.video_name, fourcc, 1, (84,84), True)
  shape = (84,84)
  frame_num = 1
  color = True 

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    y_0 = tf.placeholder(tf.float32, (1, 512))
    # unwrap it
    keep_prob = tf.placeholder("float")
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
      print("no chekcpoint file found from " + FLAGS.checkpoint_dir + FLAGS.model + FLAGS.system + ", this is an error")

    absorbing_states = []

    for _ in xrange(FLAGS.num_runs):
      # random frame
      start = np.random.rand(1, 512)

      # eval ounce
      if FLAGS.model in ("lstm_28x28x4", "lstm_84x84x4", "lstm_84x84x3"):
        generated_t_y_1, generated_t_hidden_state_1 = sess.run([y_1, hidden_state_1],feed_dict={keep_prob:1.0, y_0:start})
      elif FLAGS.model in ("fully_connected_28x28x4", "fully_connected_84x84x4", "fully_connected_84x84x3"):
        generated_t_y_1 = sess.run([y_1],feed_dict={keep_prob:1.0, y_0:start})
    
      # make video
      for step in xrange(FLAGS.run_length-1):
        # calc generated frame from t
        if FLAGS.model in ("lstm_28x28x4", "lstm_84x84x4", "lstm_84x84x3"):
          generated_t_y_1, generated_t_hidden_state_1 = sess.run([y_2, hidden_state_2],feed_dict={keep_prob:1.0, y_1:generated_t_y_1, hidden_state_1:generated_t_hidden_state_1})
        elif FLAGS.model in ("fully_connected_28x28x4", "fully_connected_84x84x4", "fully_connected_84x84x3"):
          generated_t_y_1 = sess.run([y_2],feed_dict={keep_prob:1.0, y_1:generated_t_y_1})
  
        # calc generated frame from t
      if FLAGS.model in ("lstm_28x28x4", "lstm_84x84x4", "lstm_84x84x3"):
        generated_t_x_1, generated_t_y_1, generated_t_hidden_state_1 = sess.run([x_2, y_2, hidden_state_2],feed_dict={keep_prob:1.0, y_1:generated_t_y_1, hidden_state_1:generated_t_hidden_state_1})
      elif FLAGS.model in ("fully_connected_28x28x4", "fully_connected_84x84x4", "fully_connected_84x84x3"):
        generated_t_x_1, generated_t_y_1 = sess.run([x_2, y_2],feed_dict={keep_prob:1.0, y_1:generated_t_y_1})
      generated_t_im = np.uint8(np.abs(generated_t_x_1[0, :, :, :]/np.amax(generated_t_x_1[0, :, :, :]) * 255))
  
      # concat all to a image
      new_im = generated_t_im
      new_im = np.uint8(new_im)

  
      # check if state is unique
      keep_it = True
      for i in xrange(len(absorbing_states)):
        distance = np.sqrt(np.sum(np.square(new_im - absorbing_states[i])))
        if distance < 400:
          keep_it = False 
      print("keep_it" + str(keep_it))
      if keep_it:
        print("found one!")
        absorbing_states.append(new_im)
      if len(absorbing_states) == 0:
        absorbing_states.append(new_im)


    print(len(absorbing_states)) 
    for i in xrange(len(absorbing_states)):
      cv2.imwrite("absorbing_state_" + str(i) + ".jpg", absorbing_states[i][:,:,0:3])
      # save image to video


  
def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
