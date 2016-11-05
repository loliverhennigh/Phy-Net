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

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()
success = video.open(FLAGS.video_name, fourcc, 1, (32*3, 32*3), True)

NUM_FRAMES = 60

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    x, params = ring_net.inputs(1, NUM_FRAMES) 
    # unwrap it
    output_f, output_t, output_g = ring_net.unwrap_generate(x, params, 5, NUM_FRAMES-5) 

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
    mean = mean[0]
    print(mean)
 
    # make video
    for step in xrange(NUM_FRAMES-1):
      print("making frames")
      # calc image from y_2
      new_im = np.concatenate((generated_seq[step, :, :, 0:3].squeeze()/np.amax(generated_seq[step, :, :, 0:3]), inputs[step,:,:,0:3].squeeze()/np.amax(inputs[step,:,:,0:3]), generated_seq[step, :, :, 0:3].squeeze() - inputs[step, :, :, 0:3].squeeze()), axis=0)
      if FLAGS.model in ("fully_connected_32x32x3", "lstm_32x32x3", "lstm_32x32x3_large"):
        new_im = np.concatenate((new_im, new_im, new_im), axis=1)
      new_im = np.uint8(np.abs(new_im * 255))
      #print(new_im)
      video.write(new_im)
      cv2.imwrite(FLAGS.checkpoint_dir + "_" + str(step) + ".jpg", new_im)
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
