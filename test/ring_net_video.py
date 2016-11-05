import math

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 


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
tf.app.flags.DEFINE_string('video_name', 'new_video_1.mp4',
                           """name of the video you are saving""")

writer = animation.writers['ffmpeg'](fps=30)

NUM_FRAMES = 40

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    x = ring_net.inputs(1, NUM_FRAMES) 
    # unwrap it
    keep_prob = tf.placeholder("float")
    output_t, output_g, output_f = ring_net.unwrap(x, keep_prob, NUM_FRAMES) 

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
    generated_seq, inputs = sess.run([output_g, x],feed_dict={keep_prob:1.0})
    generated_seq = generated_seq[0]
    inputs = inputs[0]
 
    # make video
    ims_generated = []
    fig = plt.figure()
    for step in xrange(NUM_FRAMES):
      # calc image from y_2
      print(step)
      new_im = np.concatenate((generated_seq[step, :, :, 2].squeeze()/np.amax(generated_seq[step, :, :, 2]), inputs[step,:,:,2].squeeze()/np.amax(inputs[step,:,:,2]), (generated_seq[step, :, :, 2].squeeze() - inputs[step,:,:,2].squeeze())), axis=0)
      ims_generated.append((plt.imshow(new_im),))
    m_ani = animation.ArtistAnimation(fig, ims_generated, interval= 5000, repeat_delay=3000, blit=True)
    print(FLAGS.video_name)
    m_ani.save(FLAGS.video_name, writer=writer)
       
def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
