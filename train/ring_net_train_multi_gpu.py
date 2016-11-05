
import os.path
import time

import numpy as np
import re
import tensorflow as tf
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt 
#import matplotlib.animation as animation 

#import Ring_Net.systems.cannon as cn
import sys
sys.path.append('../')
import systems.cannon as cn
import model.ring_net as ring_net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../checkpoints/train_store_',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """number of gpus to train on""")


#CURRICULUM_STEPS = [200, 150000, 200000, 400000]
CURRICULUM_STEPS = [300000, 150000, 200000, 200000, 800000]
CURRICULUM_SEQ = [1, 4, 6, 12, 24]
CURRICULUM_BATCH_SIZE = [4, 4, 3, 2, 2]
CURRICULUM_LEARNING_RATE = [2e-5, 2e-6, 2e-6, 1e-6, 5e-7]
#CURRICULUM_LEARNING_RATE = [5e-5, 2e-6, 2e-6, 1e-6]

def tower_loss(iteration, scope, input_keep_prob, keep_prob):
  # make inputs
  x = ring_net.inputs(CURRICULUM_BATCH_SIZE[iteration], CURRICULUM_SEQ[iteration]) 
  # possible input dropout 
  x_drop = tf.nn.dropout(x, input_keep_prob)
  # create and unrap network
  output_t, output_g, output_f = ring_net.unwrap(x_drop, keep_prob, CURRICULUM_SEQ[iteration]) 
  # calc error
  _ = ring_net.loss(x, output_t, output_g, output_f)
  # error = tf.div(error, CURRICULUM_SEQ[iteration])
  losses = tf.get_collection('losses', scope)
  # calc total loss
  total_loss = tf.add_n(losses, name='total_loss')
  # Compute the moving average of all individual losses and the total loss.
  #loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  #loss_averages_op = loss_averages.apply(losses + [total_loss])
  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  #for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
  #  loss_name = re.sub('%s_[0-9]*/' % 'tower', '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
  #  tf.scalar_summary(loss_name +' (raw)', l)
  #  tf.scalar_summary(loss_name, loss_averages.average(l))

  #with tf.control_dependencies([loss_averages_op]):
  #  total_loss = tf.identity(total_loss)

  return total_loss

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads): 
    grads = []
    for g, _ in grad_and_vars:
      if g is not None:
        expanded_g = tf.expand_dims(g, 0) 
        grads.append(expanded_g)
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def train(iteration):
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # dropout prob
    input_keep_prob = tf.placeholder("float")
    keep_prob = tf.placeholder("float")
    # global step
    global_step = tf.get_variable(
      'global_step', [],
      initializer=tf.constant_initializer(0), trainable=False)
 
    # train hopefuly 
    opt = tf.train.AdamOptimizer(CURRICULUM_LEARNING_RATE[iteration])
    #opt = tf.train.MomentumOptimizer(CURRICULUM_LEARNING_RATE[iteration], .9)
    # tower grads
    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % ('tower', i)) as scope:
          # calc error
          error = tower_loss(iteration, scope, input_keep_prob, keep_prob)
          # reuse for next tower (this needs to be checked!!!!)
          tf.get_variable_scope().reuse_variables()
          # Retain summaries for final tower
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
          # calc grads
          grads = opt.compute_gradients(error)
          # keep track of gradients
          tower_grads.append(grads)
          print("tower " + str(i) + " done!")

    # mean of gradients
    grads = average_gradients(tower_grads)

    # apply the gradients 
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

 
    # List of all Variables
    variables = tf.all_variables()

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   
    for i, variable in enumerate(variables):
      #print '----------------------------------------------'
      print variable.name[:variable.name.index(':')]

    # Summary op
    summary_op = tf.merge_summary(summaries)
 
    # Build an initialization operation to run below.
    if iteration == 0:
      init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
       allow_soft_placement=True,
       log_device_placement=False))

    # init if this is the very time training
    if iteration == 0: 
      print("init network from scratch")
      sess.run(init)

    # restore if iteration is not 0
    if iteration != 0:
      variables_to_restore = tf.all_variables()
      autoencoder_variables = [variable for i, variable in enumerate(variables_to_restore) if ("RNNCell" not in variable.name[:variable.name.index(':')]) and "tower" not in variable.name[:variable.name.index(':')]]
      rnn_variables = [variable for i, variable in enumerate(variables_to_restore) if "RNNCell" in variable.name[:variable.name.index(':')]]
      tower_variables = [variable for i, variable in enumerate(variables_to_restore) if "Merge" in variable.name[:variable.name.index(':')]]
     
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.model + FLAGS.system)
      autoencoder_saver = tf.train.Saver(autoencoder_variables)
      print("restoring autoencoder part of network form " + ckpt.model_checkpoint_path)
      autoencoder_saver.restore(sess, ckpt.model_checkpoint_path)
      print("restored file from " + ckpt.model_checkpoint_path)

      print("init tower part of network form scratch")
      for i, variable in enumerate(tower_variables):
        #print '----------------------------------------------'
        print variable.name[:variable.name.index(':')]
      #tower_init = tf.initalize_variables(tower_variables)
      #sess.run(tower_init)

      if CURRICULUM_SEQ[iteration-1] == 1:
        print("init rnn part of network from scratch")
        rnn_init = tf.initialize_variables(rnn_variables)
        sess.run(rnn_init)
      else:
        rnn_saver = tf.train.Saver(rnn_variables)
        print("restoring rnn part of network")
        rnn_saver.restore(sess, ckpt.model_checkpoint_path)
        print("restored file from " + ckpt.model_checkpoint_path)
        
    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir + FLAGS.model + FLAGS.system, graph_def=graph_def)

    for step in xrange(CURRICULUM_STEPS[iteration]):
      t = time.time()
      _ , loss_value = sess.run([train_op, error],feed_dict={keep_prob:0.9, input_keep_prob:.8})
      #print("loss value at " + str(loss_value))
      elapsed = time.time() - t

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step%100 == 0:
        summary_str = sess.run(summary_op, feed_dict={keep_prob:0.9, input_keep_prob:.8})
        summary_writer.add_summary(summary_str, step) 

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir + FLAGS.model + FLAGS.system, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir + FLAGS.model + FLAGS.system)
        print("loss value at " + str(loss_value))
        print("time per batch is " + str(elapsed))

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir + FLAGS.model + FLAGS.system):
    tf.gfile.DeleteRecursively(FLAGS.train_dir + FLAGS.model + FLAGS.system)
  tf.gfile.MakeDirs(FLAGS.train_dir + FLAGS.model + FLAGS.system)
  for i in xrange(len(CURRICULUM_STEPS)):
    train(i)

if __name__ == '__main__':
  tf.app.run()
