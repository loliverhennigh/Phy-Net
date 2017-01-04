
import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
import model.ring_net as ring_net

FLAGS = tf.app.flags.FLAGS
 
# set params for ball train
model = 'lstm_32x32x1'
system = 'diffusion'
unroll_length = 10
batch_size = 128

# save file name
SAVE_DIR = '../checkpoints/' + model + '_' + system + '_paper_' + 'seq_length_1'


def train():
  """Train ring_net for a number of steps."""
  # set flags (needs to be taken out)
  FLAGS.model = model
  FLAGS.system = system

  with tf.Graph().as_default():
    # make inputs
    print(FLAGS.system)
    state = ring_net.inputs(batch_size, unroll_length) 

    # z value
    z = tf.placeholder("float", [None, unroll_length, 10])

    # possible input dropout 
    input_keep_prob = tf.placeholder("float")
    state_drop = tf.nn.dropout(state, input_keep_prob)

    # possible dropout inside
    keep_prob_encoding = tf.placeholder("float")
    keep_prob_lstm = tf.placeholder("float")
    keep_prob_discriminator = tf.placeholder("float")

    ##### unwrap network #####
    # first step
    x_2_o = []
    x_2, hidden_state = ring_net.encode_compress_decode_gan(state[:,0,:,:,:], None, z[:,0,:], keep_prob_encoding, keep_prob_lstm)

    # discriminator with true values
    gan_t_label, gan_t_hidden_state = ring_net.discriminator(state[:,0,:,:,:], None, keep_prob_discriminator)

    # now set reuse to true
    tf.get_variable_scope().reuse_variables()

    # discriminator with generated values (start with true state and then continue with generated)
    gan_g_label, gan_g_hidden_state = ring_net.discriminator(state[:,0,:,:,:], None, keep_prob_discriminator)

    # unroll for 4 more steps on true data
    for i in xrange(int(unroll_length/2)-1):
      # unroll generator network 
      x_2, hidden_state = ring_net.encode_compress_decode_gan(state[:,i+1,:,:,:], hidden_state, z[:,i+1,:], keep_prob_encoding, keep_prob_lstm)
     
      # unroll discriminator network on true
      gan_t_label, gan_t_hidden_state = ring_net.discriminator(state[:,i+1,:,:,:], gan_t_hidden_state, keep_prob_discriminator)

      # unroll discriminator network on generated
      gan_g_label, gan_g_hidden_state = ring_net.discriminator(state[:,i+1,:,:,:], gan_g_hidden_state, keep_prob_discriminator)

    x_2_o.append(x_2) # this will be the first generated frame in the sequence
 
    # unroll discriminator network on true
    gan_t_label, gan_t_hidden_state = ring_net.discriminator(state[:,int(unroll_length/2),:,:,:], gan_t_hidden_state, keep_prob_discriminator)

    # unroll discriminator network on generated
    gan_g_label, gan_g_hidden_state = ring_net.discriminator(x_2, gan_g_hidden_state, keep_prob_discriminator)

    # now collect values
    for i in xrange(int(unroll_length/2)-1):
      # unroll generator network 
      x_2, hidden_state = ring_net.encode_compress_decode_gan(state[:,i+int(unroll_length/2),:,:,:], hidden_state, z[:,i+int(unroll_length/2),:], keep_prob_encoding, keep_prob_lstm)
      x_2_o.append(x_2)
 
      # unroll discriminator network on true
      gan_t_label, gan_t_hidden_state = ring_net.discriminator(state[:,i+int(unroll_length/2)+1,:,:,:], gan_t_hidden_state, keep_prob_discriminator)

      # unroll discriminator network on generated
      gan_g_label, gan_g_hidden_state = ring_net.discriminator(x_2, gan_g_hidden_state, keep_prob_discriminator)

      # image summary of generated
      tf.image_summary('images_gen_' + str(i), x_2)

    # reshape generated sequence
    x_2_o = tf.pack(x_2_o)
    x_2_o = tf.transpose(x_2_o, perm=[1,0,2,3,4])

    # error reconstruction
    error_reconstruction = tf.nn.l2_loss(state[:,int(unroll_length/2):,:,:,:] - x_2_o)
    tf.scalar_summary('reconstruction_loss', error_reconstruction)

    # error true
    error_d_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gan_t_label, tf.ones_like(gan_t_label))) 
    error_d_generated = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gan_g_label, tf.zeros_like(gan_g_label))) 
    error_d = error_d_true + error_d_generated
    tf.scalar_summary('error discriminator true', error_d_true)
    tf.scalar_summary('error discriminator generated', error_d_generated)
    tf.scalar_summary('error discriminator', error_d)
 
    # error generated
    error_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gan_g_label, tf.ones_like(gan_g_label))) 
    tf.scalar_summary('error generated', error_g)

    # split up variable lists
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if "discriminator" in var.name] 
    g_vars = [var for var in t_vars if "discriminator" not in var.name] 

    # make optimizers
    d_optim = tf.train.AdamOptimizer(3e-5).minimize(error_d, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(3e-5).minimize(error_g, var_list=g_vars)
    r_optim = tf.train.AdamOptimizer(1e-4).minimize(error_reconstruction)

    # List of all Variables
    variables = tf.all_variables()

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   

    # Summary op
    summary_op = tf.merge_all_summaries()
 
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(SAVE_DIR, graph_def=graph_def)

    for step in xrange(400000):
      # sample z
      z_sample = np.random.normal(0.0,1, size=(batch_size, unroll_length, 10))

      # run train step and record time
      t = time.time()
      #_, _,  loss_reconstruction, loss_g, loss_d = sess \
      #    .run([r_optim, d_optim, error_reconstruction, error_g, error_d],
      #    feed_dict={z:z_sample, keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:1.0})
      #_,  loss_reconstruction, loss_g, loss_d = sess \
      #    .run([d_optim, error_reconstruction, error_g, error_d],
      #    feed_dict={z:z_sample, keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:1.0})
      #_, loss_reconstruction, loss_g, loss_d = sess \
      #    .run([d_optim, error_reconstruction, error_g, error_d],
      #    feed_dict={z:z_sample, keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:1.0})
      #_, loss_reconstruction, loss_g, loss_d = sess \
      #    .run([g_optim, error_reconstruction, error_g, error_d],
      #    feed_dict={z:z_sample, keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:1.0})
      #loss_g, loss_d = sess \
      #    .run([error_g, error_d],
      #    feed_dict={z:z_sample, keep_prob_encoding:0.5, keep_prob_lstm:0.5, input_keep_prob:1.0, keep_prob_discriminator:.5})
      #if loss_g > loss_d:
      if step > 40000:
        _, loss_reconstruction, loss_g, loss_d = sess \
            .run([g_optim, error_reconstruction, error_g, error_d],
            feed_dict={z:z_sample, keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:1.0, keep_prob_discriminator:.5})
      #else:
      if True:
        _, loss_reconstruction, loss_g, loss_d = sess \
            .run([d_optim, error_reconstruction, error_g, error_d],
            feed_dict={z:z_sample, keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:1.0, keep_prob_discriminator:.5})
        #_, loss_reconstruction, loss_g, loss_d = sess \
        #    .run([d_optim, error_reconstruction, error_g, error_d],
        #    feed_dict={z:z_sample, keep_prob_encoding:0.5, keep_prob_lstm:0.5, input_keep_prob:1.0})
      if step < 20000:
        _, loss_reconstruction, loss_g, loss_d = sess \
            .run([r_optim, error_reconstruction, error_g, error_d],
            feed_dict={z:z_sample, keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:1.0, keep_prob_discriminator:.5})

      elapsed = time.time() - t

      assert not np.isnan(loss_reconstruction), 'Model diverged with loss reconstruction = NaN'
      assert not np.isnan(loss_g), 'Model diverged with loss generative = NaN'
      assert not np.isnan(loss_d), 'Model diverged with loss discriminator = NaN'

      if step%100 == 0:
        print("loss reconstruction " + str(loss_reconstruction))
        print("loss generated " + str(loss_g))
        print("loss discriminator " + str(loss_d))
        print("time per batch is " + str(elapsed))
        summary_str = sess.run(summary_op, feed_dict={z:z_sample, keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:1.0, keep_prob_discriminator:.5})
        summary_writer.add_summary(summary_str, step) 

      if step%1000 == 0:
        checkpoint_path = os.path.join(SAVE_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + SAVE_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(SAVE_DIR):
    tf.gfile.DeleteRecursively(SAVE_DIR)
  tf.gfile.MakeDirs(SAVE_DIR)
  train()

if __name__ == '__main__':
  tf.app.run()
