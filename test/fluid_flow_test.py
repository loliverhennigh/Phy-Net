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
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/train_store_',
                           """Directory where to read model checkpoints.""")

FLAGS.model="lstm_32x32x9"
FLAGS.system="fluid"

SECOND_SEQ = 100 
THIRD_SEQ = 200 

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    x, blocks = ring_net.inputs(1, 1) 

    # unwrap it
    output_f_mean, output_f_stddev, output_t_mean, output_t_stddev, output_g = ring_net.unwrap_3_skip(x, blocks, SECOND_SEQ, THIRD_SEQ) 

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
    generated_seq, inputs, BLOCKS = sess.run([output_g, x, blocks],feed_dict={})
    generated_seq = generated_seq[0]
    inputs = inputs[0]
    print(BLOCKS)
 
    # make video
    ims = []
    ims_generated = []
    #fig_generated = plt.figure()
    fig = plt.figure()
    for step in xrange(NUM_FRAMES):
      print(step)
      # create grid
      y, x = np.mgrid[0:1:32j, 0:1:32j]
      y, x = np.mgrid[0:1:32j, 0:1:32j]

      # create arrows for real 
      density = np.sum(inputs[step, :, :, :], axis=2)
      ux = inputs[step, :, :, 0] + inputs[step, :, :, 1] + inputs[step, :, :, 7] - inputs[step, :, :, 3] - inputs[step, :, :, 4] - inputs[step, :, :, 5]
      ux = np.divide(ux, density)
      uy = inputs[step, :, :, 1] + inputs[step, :, :, 2] + inputs[step, :, :, 3] - inputs[step, :, :, 5] - inputs[step, :, :, 6] - inputs[step, :, :, 7]
      uy = np.divide(uy, density)
      ux = np.multiply(BLOCKS, ux)
      uy = np.multiply(BLOCKS, uy)
      

      # create arrows for generated
      density_generated = np.sum(generated_seq[step, :, :, :], axis=2)
      ux_generated = generated_seq[step, :, :, 0] + generated_seq[step, :, :, 1] + generated_seq[step, :, :, 7] - generated_seq[step, :, :, 3] - generated_seq[step, :, :, 4] - generated_seq[step, :, :, 5]
      ux_generated = np.divide(ux_generated, density_generated)
      uy_generated = generated_seq[step, :, :, 1] + generated_seq[step, :, :, 2] + generated_seq[step, :, :, 3] - generated_seq[step, :, :, 5] - generated_seq[step, :, :, 6] - generated_seq[step, :, :, 7]
      uy_generated = np.divide(uy_generated, density_generated)
      #ux_generated = np.log(ux_generated)
      #uy_generated = np.log(uy_generated)
      
      print(np.max(ux_generated))
      print(np.min(ux_generated))
      print(np.max(uy_generated))

      # try to plot
      #quiver_plot_generated = plt.quiver(x, y, ux_generated, uy_generated, 
      #     color='Teal', 
      #     headlength=7)
      quiver_plot = plt.quiver(x, y, ux_generated, uy_generated, 
           color='Teal', 
           headlength=7)
      #plot2 = plt.figure()
      #plt.quiver(x, y, ux, uy, 
      #     color='Teal', 
      #     headlength=7)
      #plt.show(plot2)

      # add plots to animation array
      #ims_generated.append((quiver_plot_generated,))
      ims.append((quiver_plot,))

    # create animation
    #m_ani_generated = animation.ArtistAnimation(fig, ims_generated, interval= 5000, repeat_delay=3000, blit=True)
    m_ani = animation.ArtistAnimation(fig, ims, interval= 5000, repeat_delay=3000, blit=True)

    # save animations
    #m_ani_generated.save("generated_" + FLAGS.video_name, writer=writer_generated)
    m_ani.save(FLAGS.video_name, writer=writer)
    print("saved to " + FLAGS.video_name + " and generated_" + FLAGS.video_name)
       
def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
