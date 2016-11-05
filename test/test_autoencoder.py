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
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/train_store_',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('video_name', 'color_video.mov',
                           """name of the video you are saving""")

assert(FLAGS.model in ("fully_connected_84x84x4", "fully_connected_84x84x3", "lstm_84x84x4", "lstm_84x84x3"), "need to use a model thats 84x84, sorry")

NUM_FRAMES = 60

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    x, params = ring_net.inputs(1, NUM_FRAMES) 
    # unwrap it
    x_1 = []
    for i in xrange(NUM_FRAMES):
      # set reuse to true
      if i > 0:
        tf.get_variable_scope().reuse_variables() 

      x_out = ring_net.autoencoder(x, i) 
      x_1.append(x_out)

    # restore network
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("restored file from " + ckpt.model_checkpoint_path)
    else:
      print("no chekcpoint file found, this is an error")

    # start que runner
    tf.train.start_queue_runners(sess=sess)
 
    # eval ounce
    generated_inputs, inputs = sess.run([x_1, x],feed_dict={})
    #generated_inputs = generated_inputs[0]
    inputs = inputs[0]

    # make video
    for step in xrange(NUM_FRAMES-1):
      print("making frames")
      # calc image from y_2
      new_im = np.concatenate((generated_inputs[step][0, :, :, 0:3].squeeze()/np.amax(generated_inputs[step][0, :, :, 0:3]), inputs[step, :,:,0:3].squeeze()/np.amax(inputs[step, :,:,0:3]), generated_inputs[step][0, :, :, 0:3].squeeze() - inputs[step, :, :, 0:3].squeeze()), axis=0)
      new_im = np.uint8(np.abs(new_im * 255))
      cv2.imwrite("test_" + str(step) + ".jpg", new_im)

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
