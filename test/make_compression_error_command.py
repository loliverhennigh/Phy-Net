import sys
sys.path.append('../')

from utils.experiment_manager import *

os.system('rm compression_error_run_command.txt')

cmd = []

# 2d sims
checkpoint_paths = list_all_checkpoints('../checkpoints_2d_compression_experiments')
for i in xrange(len(checkpoint_paths)):
  flags = make_flags_string_given_checkpoint_path(checkpoint_paths[i]) + ' --base_dir=../checkpoints_2d_compression_experiments'
  cmd.append('python generate_compression_error_plot.py' + flags) 

# 3d sims
checkpoint_paths = list_all_checkpoints('../checkpoints_3d_compression_experiments')
for i in xrange(len(checkpoint_paths)):
  flags = make_flags_string_given_checkpoint_path(checkpoint_paths[i]) + ' --base_dir=../checkpoints_3d_compression_experiments'
  cmd.append('python generate_compression_error_plot.py' + flags) 

with open("compression_error_run_command.txt", "a") as myfile:
  for c in cmd:
    myfile.write(c + '\n')
   



