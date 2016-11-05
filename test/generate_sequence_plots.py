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
tf.app.flags.DEFINE_string('input_video', '../data/video/slkdf',
                           """name of the video you are saving""")
tf.app.flags.DEFINE_string('video_name', 'color_video.mov',
                           """name of the video you are saving""")
tf.app.flags.DEFINE_integer('run_length', 1000,
                           """number of frames to run out""")
tf.app.flags.DEFINE_integer('start_frame', 2000,
                           """number of frames to run out""")

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()
error_t = []
error_f =[]
error_tf =[]

if FLAGS.model in ("fully_connectd_28x28x4", "lstm_28x28x4"):
  success = video.open(FLAGS.video_name, fourcc, 4, (28, 84), True)
  shape = (28,28)
  frame_num = 4
  color = False
elif FLAGS.model in ("fully_connected_84x84x4", "lstm_84x84x4"):
  #success = video.open(FLAGS.video_name, fourcc, 4, (252, 84), True)
  success = video.open(FLAGS.video_name, fourcc, 4, (84,252), True)
  shape = (84,84)
  frame_num = 4
  color = False
elif FLAGS.model in ("fully_connected_84x84x3", "lstm_84x84x3"):
  #success = video.open(FLAGS.video_name, fourcc, 4, (252, 84), True)
  success = video.open(FLAGS.video_name, fourcc, 4, (84,252), True)
  shape = (84,84)
  frame_num = 1
  color = True 

def get_seq_frame_test(cap, seq_length, frame_num, shape, color):
  # the stored frames
  if color:
    frames = np.zeros((shape[0], shape[1], frame_num*3))
    seq_frames = np.zeros((seq_length, shape[0], shape[1], frame_num*3))
  else:
    frames = np.zeros((shape[0], shape[1], frame_num))
    seq_frames = np.zeros((seq_length, shape[0], shape[1], frame_num))

  # num frames
  ind = 0

  # create frames
  for s in xrange(seq_length):
    if ind == 0:
      for i in xrange(frame_num):
        if color:
          frames[:,:,i*3:(i+1)*3] = get_converted_frame(cap, shape, color)
        else:
          frames[:,:,i] = get_converted_frame(cap, shape, color)

      ind = ind + 1
    else:
      if color:
        frames[:,:,0:frame_num*3-3] = frames[:,:,3:frame_num*3]
        frames[:,:,(frame_num-1)*3:frame_num*3] = get_converted_frame(cap, shape, color)

      else:
        frames[:,:,0:frame_num-1] = frames[:,:,1:frame_num]
        frames[:,:,frame_num-1] = get_converted_frame(cap, shape, color)
 
    seq_frames[s, :, :, :] = frames[:,:,:]
  
  return np.uint8(seq_frames)


def get_converted_frame(cap, shape, color):
  for i in xrange(FLAGS.video_frames_per_train_frame):
    ret, frame = cap.read()
  frame = cv2.resize(frame, shape, interpolation = cv2.INTER_CUBIC)
  if color:
    return frame
  else:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make video cap
    cap = cv2.VideoCapture(FLAGS.input_video) 
    for _ in xrange(FLAGS.start_frame):
      ret, frame = cap.read()
    # make inputs
    if color:
      image = tf.placeholder(tf.uint8, (1, 1, shape[0], shape[1], 3*frame_num))
    else:
      image = tf.placeholder(tf.uint8, (1, 1, shape[0], shape[1], frame_num))
    x = tf.to_float(image)
    x = tf.div(x, 255.0)
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
      print("no chekcpoint file found from " + FLAGS.checkpoint_dir + FLAGS.model + FLAGS.system + ", this is an error")

    # get frame 
    seq_frame = get_seq_frame_test(cap, FLAGS.run_length, frame_num, shape, color)
    frame = np.expand_dims(seq_frame[0, :, :, :], axis=0)
    frame = np.expand_dims(frame, axis=0)

    # eval ounce
    if FLAGS.model in ("lstm_28x28x4", "lstm_84x84x4", "lstm_84x84x3"):
      generated_t_x_1, generated_t_y_1, generated_t_hidden_state_1 = sess.run([x_1, y_1, hidden_state_1],feed_dict={keep_prob:1.0, image:frame})
    elif FLAGS.model in ("fully_connected_28x28x4", "fully_connected_84x84x4", "fully_connected_84x84x3"):
      generated_t_x_1, generated_t_y_1 = sess.run([x_1, y_1],feed_dict={keep_prob:1.0, image:frame})
    generated_t_im = np.uint8(np.abs(generated_t_x_1[0, :, :, :]/np.amax(generated_t_x_1[0, :, :, :]) * 255))
    generated_f_x_1 = np.copy(generated_t_x_1)
    generated_f_y_1 = np.copy(generated_t_y_1)
    generated_f_hidden_state_1 = generated_t_hidden_state_1.copy()
    
    # append error
    error_t.append(np.sqrt(np.sum(np.square(generated_t_x_1[0, :, :, :] - (seq_frame[0, :, :, :]/255.0)))))
    error_f.append(np.sqrt(np.sum(np.square(generated_f_x_1[0, :, :, :] - (seq_frame[0, :, :, :]/255.0)))))
    error_tf.append(np.sqrt(np.sum(np.square(generated_t_x_1[0, :, :, :] - generated_f_x_1[0, :, :, :]))))
    
    # concat frames top t, middle full, bottum true 
    new_im = np.concatenate((generated_t_im, generated_t_im, seq_frame[0, :, :, :]), axis=0)
    new_im = np.uint8(new_im)
    video.write(new_im[:,:,0:3])
 
    # make video
    for step in xrange(FLAGS.run_length-1):
      # continue to calc frames
      #print(step)
     
      # calc generated frame from t
      if FLAGS.model in ("lstm_28x28x4", "lstm_84x84x4", "lstm_84x84x3"):
        generated_t_x_1, generated_t_y_1, generated_t_hidden_state_1 = sess.run([x_2, y_2, hidden_state_2],feed_dict={keep_prob:1.0, y_1:generated_t_y_1, hidden_state_1:generated_t_hidden_state_1})
      elif FLAGS.model in ("fully_connected_28x28x4", "fully_connected_84x84x4", "fully_connected_84x84x3"):
        generated_t_x_1, generated_t_y_1 = sess.run([x_2, y_2],feed_dict={keep_prob:1.0, y_1:generated_t_y_1})
      generated_t_im = np.uint8(np.abs(generated_t_x_1[0, :, :, :]/np.amax(generated_t_x_1[0, :, :, :]) * 255))

      # calc generated frames from full network
      if FLAGS.model in ("lstm_28x28x4", "lstm_84x84x4", "lstm_84x84x3"):
        generated_f_x_1, generated_f_y_1, generated_f_hidden_state_1 = sess.run([x_2, y_2, hidden_state_2],feed_dict={keep_prob:1.0, y_1:generated_f_y_1, hidden_state_1:generated_f_hidden_state_1})
      elif FLAGS.model in ("fully_connected_28x28x4", "fully_connected_84x84x4", "fully_connected_84x84x3"):
        generated_f_x_1, generated_f_y_1 = sess.run([x_2, y_2],feed_dict={keep_prob:1.0, y_1:generated_f_y_1})
      generated_f_x_1 = np.expand_dims(generated_f_x_1, axis=0)
      generated_f_y_1 = sess.run([y_0], feed_dict={keep_prob:1.0, x:generated_f_x_1})
      generated_f_y_1 = generated_f_y_1[0] 
      generated_f_im = np.uint8(np.abs(generated_f_x_1[0, 0, :, :, :]/np.amax(generated_f_x_1[0, 0, :, :, :]) * 255))
      
      # concat all to a image
      new_im = np.concatenate((generated_t_im, generated_f_im, seq_frame[step+1, :, :, :]), axis=0)
      new_im = np.uint8(new_im)

      # calc error
      error_t.append(np.sqrt(np.sum(np.square(generated_t_x_1[0, :, :, :] - (seq_frame[step+1, :, :, :]/255.0)))))
      error_f.append(np.sqrt(np.sum(np.square(generated_f_x_1[0, :, :, :] - (seq_frame[step+1, :, :, :]/255.0)))))
      error_tf.append(np.sqrt(np.sum(np.square(generated_t_x_1[0, :, :, :] - generated_f_x_1[0, :, :, :]))))

      # save image to video
      video.write(new_im[:,:,0:3])

    print('saved to ' + FLAGS.video_name)
    video.release()
    cv2.destroyAllWindows()

    plt.plot(error_t, label="error_t")
    plt.plot(error_f, label="error_f")
    plt.plot(error_tf, label="error_tf")
    plt.legend()
    plt.show()
       
def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
