
import sys
sys.path.append('../')
import utils.que as que

q = que.Que([0])
q.enque_file("2d_experiments.txt")
q.start_que_runner()





