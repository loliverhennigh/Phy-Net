
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

with open("figs/compression_error_log.txt", "r") as myfile:
  data = myfile.readlines()
data = [x.strip() for x in data]

mse_2d = [] 
compression_factor_2d = []
mse_3d = []
compression_factor_3d = []

for i in xrange(len(data)):
  seperated_line = data[i].split()
  if int(seperated_line[0]) == 2:
    mse_2d.append(float(seperated_line[1]))
    compression_factor_2d.append(float(seperated_line[2]))
  elif int(seperated_line[0]) == 3:
    mse_3d[i].append(float(seperated_line[1]))
    compression_factor_3d[i].append(float(seperated_line[2]))

mse_2d = np.array(mse_2d)
compression_factor_2d = np.array(compression_factor_2d)
mse_3d = np.array(mse_3d)
compression_factor_3d = np.array(compression_factor_3d)

plt.style.use('seaborn-darkgrid')

font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 6}

matplotlib.rc('font', **font)

plt.scatter(compression_factor_2d, mse_2d)
plt.scatter(compression_factor_3d, mse_3d)

plt.title('Compression Vs Error', y=0.96)
plt.xlabel('Compression Ratio')
plt.ylabel('Mean Squared Error')
plt.legend(loc="upper_left")
plt.savefig("figs/compression_error_plot.png")

plt.show()




