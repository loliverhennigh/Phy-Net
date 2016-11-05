import math

import numpy as np
import tensorflow as tf


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

assert(FLAGS.model in ("fully_connected_84x84x4", "fully_connected_84x84x3", "lstm_84x84x4", "lstm_84x84x3"), "need to use a model thats 84x84, sorry")

SECOND_SEQ = 100 
THIRD_SEQ = 200 

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    x, params = ring_net.inputs(1, 1) 
    # unwrap it
    output_f_mean, output_f_stddev, output_t_mean, output_t_stddev, output_g = ring_net.unwrap_generate_3_skip(x, params, SECOND_SEQ, THIRD_SEQ) 

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
    print("generated_seq")
    generated_seq, inputs = sess.run([output_g, x],feed_dict={})
    generated_seq = generated_seq[0]
    inputs = inputs[0]
 
    # make video
    for step in xrange(9):
      print("making frames")
      # calc image from y_2
      new_im = generated_seq[step, :, :, 0:3].squeeze()/np.amax(generated_seq[step, :, :, 0:3])
      new_im = np.uint8(np.abs(new_im * 255))
      #print(new_im)
      cv2.imwrite("long_unravel_images_" + str(step) + ".jpg", new_im)
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
