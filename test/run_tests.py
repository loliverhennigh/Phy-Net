
import sys
sys.path.append('../')
import utils.que as que

q = que.Que([-1])
#q = que.Que([0,1,2])
q.enque_file("2d_error_test.txt")
q.enque_file("3d_error_test.txt")
q.enque_file("2d_video_test.txt")
q.enque_file("3d_video_test.txt")
q.enque_file("compression_error_run_command.txt")
q.start_que_runner()





