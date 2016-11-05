import math

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 


import sys
sys.path.append('../')
import model.ring_net as  ring_net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../checkpoints/ring_net_eval_store',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/train_store_',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('video_name', 'new_video_1.mp4',
                           """name of the video you are saving""")

writer = animation.writers['ffmpeg'](fps=30)

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # define net
    y = tf.placeholder(tf.float32, [None, 512])
    x = ring_net.decoding(y) 

    # restore network
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir + FLAGS.model + FLAGS.system)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("restored file from " + ckpt.model_checkpoint_path)
    else:
      print("no chekcpoint file found, this is an error")
 
    # make video
    y_state = np.zeros((1,512))
    ims_generated = []
    fig = plt.figure()
    for step in xrange(512):
      # calc image from y_2
      print(step)
      y_state = np.zeros((1,512))
      y_state[0, step] = 1.0
      generated_x = x.eval(session=sess,feed_dict={y:y_state})
      generated_x = generated_x[0, :, :, 0].reshape((28,28))
      ims_generated.append((plt.imshow(generated_x),))
    m_ani = animation.ArtistAnimation(fig, ims_generated, interval= 5000, repeat_delay=3000, blit=True)
    print(FLAGS.video_name)
    m_ani.save(FLAGS.video_name, writer=writer)
       
def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.model != 'markov_28x28x4':
    raise ValueError('this only works for markov networks ')
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
