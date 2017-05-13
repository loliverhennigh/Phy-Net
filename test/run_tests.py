
import sys
sys.path.append('../')
import utils.que as que

q = que.Que([0])
"""
q.enque_file("2d_error_test.txt")
q.enque_file("2d_image_test.txt")
q.enque_file("2d_video_test.txt")
q.enque_file("car_error_test.txt")
q.enque_file("car_image_test.txt")
q.enque_file("car_video_test.txt")
q.enque_file("em_error_test.txt")
q.enque_file("em_image_test.txt")
q.enque_file("em_video_test.txt")
q.enque_file("3d_error_test.txt")
q.enque_file("3d_image_test.txt")
q.enque_file("3d_video_test.txt")
"""
q.enque_file("compression_error_run_command.txt")
q.start_que_runner()





