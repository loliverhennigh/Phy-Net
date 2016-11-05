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

tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/train_store_',
                           """Directory where to read model checkpoints.""")

NUM_FRAMES = 100

def evaluate():
  # set default flags
  FLAGS.model="lstm_32x32x10"
  FLAGS.system="fluid"

  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    x = ring_net.inputs(1, NUM_FRAMES) 

    # unwrap it
    output_f_mean, output_f_stddev, output_t_mean, output_t_stddev, output_g = ring_net.unwrap_generate(x, 5, NUM_FRAMES-5)

    # restore network
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    #ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("restored file from " + ckpt.model_checkpoint_path)
    else:
      print("no chekcpoint file found, this is an error")

    # start que runner
    tf.train.start_queue_runners(sess=sess)

    # eval ounce
    generated_inputs, inputs = sess.run([output_g, x],feed_dict={})
    generated_inputs = generated_inputs[0]
    inputs = inputs[0]

    # create grid
    y_p, x_p = np.mgrid[0:1:32j, 0:1:32j]


    for i in xrange(NUM_FRAMES):
      # create arrows for real 
      density = np.sum(inputs[i, :, :, :], axis=2)
      ux = inputs[i, :, :, 0] + inputs[i, :, :, 1] + inputs[i, :, :, 7] - inputs[i, :, :, 3] - inputs[i, :, :, 4] - inputs[i, :, :, 5]
      ux = np.divide(ux, density)
      uy = inputs[i, :, :, 1] + inputs[i, :, :, 2] + inputs[i, :, :, 3] - inputs[i, :, :, 5] - inputs[i, :, :, 6] - inputs[i, :, :, 7]
      uy = np.divide(uy, density)
      print(ux.shape)
  
      # create arrows for generated 
      generated_density = np.sum(generated_inputs[i, :, :, :], axis=2)
      generated_ux = generated_inputs[i, :, :, 0] + generated_inputs[i, :, :, 1] + generated_inputs[i, :, :, 7] - generated_inputs[i, :, :, 3] - generated_inputs[i, :, :, 4] - generated_inputs[i, :, :, 5]
      generated_ux = np.divide(generated_ux, generated_density)
      generated_uy = generated_inputs[i, :, :, 1] + generated_inputs[i, :, :, 2] + generated_inputs[i, :, :, 3] - generated_inputs[i, :, :, 5] - generated_inputs[i, :, :, 6] - generated_inputs[i, :, :, 7]
      generated_uy = np.divide(generated_uy, generated_density)
      print(generated_ux.shape)
  
      # make blocks
      '''BLOCKS = np.zeros((32, 32)) + 1.0
      BLOCKS[:, np.floor(32.0/4)-1] = 0.0
      BLOCKS[:, 2*np.floor(32.0/4)-1] = 0.0
      BLOCKS[:, 3*np.floor(32.0/4)-1] = 0.0
      BLOCKS[blocks[i,0]:blocks[i,0]+10, np.floor(32.0/4)-1] = 1.0
      BLOCKS[blocks[i,1]:blocks[i,1]+10, 2*np.floor(32.0/4)-1] = 1.0
      BLOCKS[blocks[i,2]:blocks[i,2]+10, 3*np.floor(32.0/4)-1] = 1.0 '''
      blocks_generated = 1.0 - generated_inputs[i,:,:,9]
      blocks = 1.0 - inputs[i,:,:,9]
  
      # kill vel real
      ux = np.multiply(blocks, ux)
      uy = np.multiply(blocks, uy)
      
      # kill vel real
      generated_ux = np.multiply(blocks_generated, generated_ux)
      generated_uy = np.multiply(blocks_generated, generated_uy)
      
      # now plot  
      plot2 = plt.figure()
      #plt.pcolor(x_p, y_p, np.square(uy - generated_uy), cmap='RdBu')
      plt.pcolor(x_p, y_p, np.square(generated_density - density), cmap='RdBu')
      #plt.pcolor(x_p, y_p, generated_density, cmap='RdBu')
      #plt.pcolor(x_p, y_p, density, cmap='RdBu')
      plt.colorbar()
      plt.quiver(x_p, y_p, ux, uy, 
             color='Teal', 
             headlength=7)

      plt.quiver(x_p, y_p, generated_ux, generated_uy, 
             color='Blue', 
             headlength=7)
      plt.show(plot2)
  
def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
