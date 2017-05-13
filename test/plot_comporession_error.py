
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
    mse_3d.append(float(seperated_line[1]))
    compression_factor_3d.append(float(seperated_line[2]))

mse_2d = np.array(mse_2d)
compression_factor_2d = np.array(compression_factor_2d)
mse_3d = np.array(mse_3d)
compression_factor_3d = np.array(compression_factor_3d)

plt.style.use('seaborn-darkgrid')

font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 16}

matplotlib.rc('font', **font)

plt.scatter(np.log(compression_factor_2d), np.log(mse_2d), color='red', label="2D")
plt.scatter(np.log(compression_factor_3d), np.log(mse_3d), color='purple', label="3D")

plt.title('Compression Vs Error', y=1.00, fontsize="x-large")
plt.xlabel(r'$\log{ \frac{compressed \  size}{state \  size}}$', fontsize="x-large", y=2.10)
plt.ylabel(r'$\log{(MSError)}$', fontsize="x-large")
plt.legend(loc="upper_left")
plt.tight_layout()
plt.savefig("figs/compression_error_plot.png")

plt.show()




