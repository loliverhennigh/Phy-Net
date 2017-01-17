

import tensorflow as tf

def apply_boundry(state, boundry):

  state = state * boundry.kill_state
  state = state + boundry.add_state

  return state

def add_boundry(state, boundry):

  len_state = len(state.get_shape())
  state = tf.concat(len_state - 1, [state, boundry.concat_state])

  return state



