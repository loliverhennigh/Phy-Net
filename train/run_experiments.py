
import sys
sys.path.append('../')
import utils.que as que

q = que.Que([0])
#q = que.Que([0,1,2]) # if you have more then one gpu put it here
q.enque_file("2d_show_case_experiments.txt")
q.enque_file("3d_show_case_experiments.txt")
q.start_que_runner()





