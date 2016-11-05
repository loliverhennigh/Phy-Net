import numpy as np
import cv2
import scipy.io
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt


fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()
success = video.open("test.mov", fourcc, 4, (32, 32), True)




T = 1500 
#mat = scipy.io.loadmat('./store_fluid_flow_test/run_0/state_step_' + str(1) + '.mat')
#bound = scipy.io.loadmat('./store_fluid_flow_test/run_0/bound.mat')
#inputs_prev = np.array(mat['F'])

steady_state = np.array([1.0/9, 1.0/36, 1.0/9, 1.0/36, 1.0/9, 1.0/36, 1.0/9, 1.0/36, 4.0/9]) 
print(np.sum(steady_state))

for i in xrange(T):
  # read in files
  mat = scipy.io.loadmat('./store_fluid_flow_size_32/run_0/state_step_' + str(i+1) + '.mat')
  bound = scipy.io.loadmat('./store_fluid_flow_size_32/run_0/bound.mat')

  # make inputs
  inputs = np.array(mat['F_train'])
  #inputs = np.array(mat['F'])
  density = np.sum(inputs[:, :, :], axis=2)
  ux = inputs[:, :, 0] + inputs[:, :, 1] + inputs[:, :, 7] - inputs[:, :, 3] - inputs[:, :, 4] - inputs[:, :, 5]
  ux = np.divide(ux, density)
  uy = inputs[:, :, 1] + inputs[:, :, 2] + inputs[:, :, 3] - inputs[:, :, 5] - inputs[:, :, 6] - inputs[:, :, 7]
  uy = np.divide(uy, density)


  #print(np.log(inputs[7,7,:] - steady_state))
  #print(inputs[7,7,:] - steady_state)
  #print(inputs[10,10,:] - steady_state)
  print(inputs[1,6,:])
  print(inputs[1,7,:])
  #print(inputs[2,7,:])
  #print(inputs[10,10,:])
  print(np.max(inputs[:,:,:]) * 100.0)
  print(np.min(inputs[:,:,:]) * 100.0)
  print(i)
  inputs_prev = inputs
  # kill bound values
  bound = np.array(bound['BOUND'], dtype=float)
  bound = (-bound) + 1.0
  print(bound[1,6])
  #print(bound)
  ux = np.multiply(bound, ux)
  uy = np.multiply(bound, uy)

  #arrows_x = np.array(mat['UX'])
  #arrows_y = np.array(mat['UY'])
  y, x = np.mgrid[0:1:32j, 0:1:32j]
  if i % 50 == 0:
    plot2 = plt.figure()
    plt.pcolor(x, y, density, cmap='RdBu')
    plt.quiver(x, y, ux, uy, 
           color='Teal', 
           headlength=7)
    plt.show(plot2)


video.release()
cv2.destroyAllWindows()
