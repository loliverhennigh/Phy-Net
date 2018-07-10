import tensorflow as tf
import fnmatch
import os

# flags to not include in checkpoint path
non_checkpoint_flags = ['min_queue_examples', 'data_dir', 'tf_data_dir', 'num_preprocess_threads', 'train', 'base_dir', 'restore', 'max_steps', 'restore_unroll_length', 'batch_size', 'unroll_from_true', 'unroll_length', 'video_shape', 'video_length', 'test_length', 'test_nr_runs', 'test_nr_per_simulation', 'test_dimensions', 'lstm', 'gan', 'nr_discriminators', 'z_size', 'nr_downsamples_discriminator', 'nr_residual_discriminator', 'keep_p_discriminator', 'filter_size_discriminator', 'lstm_size_discriminator', 'lambda_reconstruction', 'nr_gpus', 'tf_store_images', 'gan_lr', 'init_unroll_length', 'tf_seq_length', 'extract_type', 'extract_pos']

def str2bool(v):
  return v == 'TRUE'

def make_checkpoint_path(base_path, FLAGS):
  # make checkpoint path with all the flags specifing different directories

  # run through all params and add them to the base path
  # run through all params and add them to the base path
  keys = FLAGS.flag_values_dict().keys()
  keys.sort()
  for k in keys:
    if k not in NOT_PATH:
      base_path = base_path + '/' + k + '.' + str(FLAGS.flag_values_dict()[k])

  return base_path

def list_all_checkpoints(base_path):
  # get a list off all the checkpoint directorys

  # run through all params and add them to the base path
  paths = []
  for root, dirnames, filenames in os.walk(base_path):
    for filename in fnmatch.filter(filenames, 'checkpoint'):
      paths.append(root[len(base_path)+1:])
  return paths

def set_flags_given_checkpoint_path(path, FLAGS):
  # get a list off all the checkpoint directorys

  # run through all params and add them to the base path
  split_path = path.split('/')
  for param in split_path:
    split_param = param.split('.')
    param_name = split_param[0]
    param_value = '.'.join(split_param[1:])
    param_type = type(FLAGS.__dict__['__flags'][param_name])
    if param_type == bool:
      param_type = str2bool
    FLAGS.__dict__['__flags'][param_name] = param_type(param_value)

def make_flags_string_given_checkpoint_path(path):
  # run through all params and add them to the base path
  flag_string = ''
  split_path = path.split('/')
  for param in split_path:
    split_param = param.split('.')
    param_name = split_param[0]
    param_value = '.'.join(split_param[1:])
    flag_string += ' --' + param_name + '=' + param_value
  return flag_string

