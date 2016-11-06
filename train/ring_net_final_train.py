
import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
import systems.cannon as cn
import model.ring_net as ring_net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../checkpoints/train_store_',
                            """dir to store trained net""")
tf.app.flags.DEFINE_float('input_keep_prob', 1.0,
                          """ input keep probability """)
tf.app.flags.DEFINE_float('keep_prob_encoding', .5,
                          """ keep encoding probability """)
tf.app.flags.DEFINE_float('keep_prob_lstm', 1.0,
                          """ keep lstm probability """)
tf.app.flags.DEFINE_float('keep_prob_decoding', 1.0,
                          """ keep decoding probability """)
tf.app.flags.DEFINE_bool('load_network', True,
                          """ whether to load the network or not """)

if FLAGS.model in ("fully_connected_32x32x3", "fully_connected_84x84x3", "fully_connected_84x84x4"):
  CURRICULUM_STEPS = [300000]
  CURRICULUM_SEQ = [20]
  CURRICULUM_BATCH_SIZE = [32]
  CURRICULUM_LEARNING_RATE = [1e-6]
  CURRICULUM_TRAIN_PIECE = ["all"]
elif FLAGS.model in ("lstm_32x32x3", "lstm_84x84x3", "lstm_84x84x4", "lstm_32x32x1", "lstm_32x32x10"):
  CURRICULUM_STEPS = [1000000]
  #CURRICULUM_STEPS = [30, 30, 10]
  CURRICULUM_SEQ = [10]
  CURRICULUM_BATCH_SIZE = [32]
  CURRICULUM_LEARNING_RATE = [1e-6]
  CURRICULUM_TRAIN_PIECE = ["all"]
  #CURRICULUM_STEPS = [50000, 50000000]
  #CURRICULUM_SEQ = [1, 20]
  #CURRICULUM_BATCH_SIZE = [64, 32]
  #CURRICULUM_LEARNING_RATE = [5e-5, 2e-6]
  #CURRICULUM_AUTOENCODER = [True, False]
  
load_dir = FLAGS.train_dir + FLAGS.model + FLAGS.system + "_compression" + FLAGS.compression_loss + "_compression_vae_loss_" + str(FLAGS.compression_vae_loss) + "_sample_compression_" + str(FLAGS.sample_compression) + "_lstm_size_" + str(FLAGS.lstm_size) + "_num_layers_" + str(FLAGS.num_layers)
train_dir = FLAGS.train_dir + FLAGS.model + FLAGS.system + "_all_" + FLAGS.compression_loss + "_compression_vae_loss_" + str(FLAGS.compression_vae_loss) + "_sample_compression_" + str(FLAGS.sample_compression) + "_lstm_size_" + str(FLAGS.lstm_size) + "_num_layers_" + str(FLAGS.num_layers)


def train(iteration):
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = ring_net.inputs(CURRICULUM_BATCH_SIZE[iteration], CURRICULUM_SEQ[iteration]) 

    # possible input dropout 
    input_keep_prob = tf.placeholder("float")
    x_drop = tf.nn.dropout(x, input_keep_prob)

    # possible dropout inside
    keep_prob_encoding = tf.placeholder("float")
    keep_prob_lstm = tf.placeholder("float")
    keep_prob_decoding = tf.placeholder("float")

    # create and unrap network
    output_f, output_t, output_g, output_autoencoding = ring_net.unwrap(x_drop, keep_prob_encoding, keep_prob_lstm, keep_prob_decoding, CURRICULUM_SEQ[iteration], CURRICULUM_TRAIN_PIECE[iteration]) 

    # calc error
    total_loss = ring_net.loss(x, output_f, output_t, output_g, output_autoencoding, CURRICULUM_TRAIN_PIECE[iteration])
    #loss_vae, loss_reconstruction_autoencoder, loss_reconstruction_g, loss_t = ring_net.loss(x, output_f_mean, output_f_stddev, output_t_mean, output_t_stddev, output_g, output_autoencoding, CURRicULUM_TRAIN_PIECE[iteration])
    #if CURRICULUM_AUTOENCODER[iteration]:
    #  total_loss = tf.reduce_mean(loss_vae + loss_reconstruction_autoencoder)
    #else:
    #  total_loss = tf.reduce_mean(loss_vae + loss_reconstruction_autoencoder + loss_reconstruction_g + loss_t)
    total_loss = tf.div(total_loss, CURRICULUM_SEQ[iteration])

    # train hopefuly 
    train_op = ring_net.train(total_loss, CURRICULUM_LEARNING_RATE[iteration])
    
    # List of all Variables
    variables = tf.all_variables()
    #for i, variable in enumerate(variables):
    #  print '----------------------------------------------'
    #  print variable.name[:variable.name.index(':')]

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   

    # Summary op
    summary_op = tf.merge_all_summaries()
 
    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    #opt_slot_var = optim.get_slot_names()
    #opt_slot_init = tf.initialize_variables(opt_slot_var)
    #sess.run(opt_slot_init)

    # restore if iteration is not 0
    if FLAGS.load_network:
      variables_to_restore = tf.all_variables()
      autoencoder_variables = [variable for i, variable in enumerate(variables_to_restore) if ("compress" not in variable.name[:variable.name.index(':')]) and ("RNN" not in variable.name[:variable.name.index(':')])]
      rnn_variables = [variable for i, variable in enumerate(variables_to_restore) if ("compress" in variable.name[:variable.name.index(':')]) or ("RNN" in variable.name[:variable.name.index(':')])]
     
      if iteration == 0: 
        ckpt_autoencoder = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.model + FLAGS.system + "_autoencoder")
        ckpt_compression = tf.train.get_checkpoint_state(load_dir)
      else:
        ckpt_autoencoder = tf.train.get_checkpoint_state(train_dir)
        ckpt_compression = tf.train.get_checkpoint_state(train_dir)

      autoencoder_saver = tf.train.Saver(autoencoder_variables)
      print("restoring autoencoder part of network form " + ckpt_autoencoder.model_checkpoint_path)
      autoencoder_saver.restore(sess, ckpt_autoencoder.model_checkpoint_path)
      print("restored file from " + ckpt_autoencoder.model_checkpoint_path)

      rnn_saver = tf.train.Saver(rnn_variables)
      print("restoring compression part of network from " + load_dir)
      rnn_saver.restore(sess, ckpt_compression.model_checkpoint_path)
      print("restored file from " + ckpt_compression.model_checkpoint_path)
        
    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(train_dir, graph_def=graph_def)

    for step in xrange(CURRICULUM_STEPS[iteration]):
      t = time.time()
      _ , loss_r = sess.run([train_op, total_loss],feed_dict={keep_prob_encoding:FLAGS.keep_prob_encoding, keep_prob_lstm:FLAGS.keep_prob_lstm, keep_prob_decoding:FLAGS.keep_prob_decoding, input_keep_prob:FLAGS.input_keep_prob})
      elapsed = time.time() - t
      #print("loss vae value at " + str(loss_vae_r))
      #print("loss value at " + str(loss_r))
      #print("g " + str(np.max(g_r)))
      #print("g " + str(np.min(g_r)))
      #print("im " + str(np.max(im)))
      #print("im " + str(np.min(im)))
      #print("time per batch is " + str(elapsed))


      if step%100 == 0:
        print("loss value at " + str(loss_r))
        #print("g " + str(np.max(g_r)))
        #print("g " + str(np.min(g_r)))
        print("time per batch is " + str(elapsed))
        summary_str = sess.run(summary_op, feed_dict={keep_prob_encoding:FLAGS.keep_prob_encoding, keep_prob_lstm:FLAGS.keep_prob_lstm, keep_prob_decoding:FLAGS.keep_prob_decoding, input_keep_prob:FLAGS.input_keep_prob})
        summary_writer.add_summary(summary_str, step) 
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + train_dir)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(train_dir):
    tf.gfile.DeleteRecursively(train_dir)
  tf.gfile.MakeDirs(train_dir)
  for i in xrange(len(CURRICULUM_STEPS)):
    train(i)

if __name__ == '__main__':
  tf.app.run()
