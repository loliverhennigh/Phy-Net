


import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import sys

show = "bounds"

if len(sys.argv) > 1:
  show = sys.argv[1]

import matplotlib.image as mpimg

def divergence(velocity_field):
  velocity_field_x_0 = velocity_field[0:-2,1:-1,1]
  velocity_field_x_2 = velocity_field[2:,1:-1,1]

  velocity_field_y_0 = velocity_field[1:-1,0:-2,0]
  velocity_field_y_2 = velocity_field[1:-1,2:,0]

  div = velocity_field_x_0 - velocity_field_x_2 + velocity_field_y_0 - velocity_field_y_2

  return div

for i in xrange(1000):
  if i % 1 == 0:
    print(i)
    name = "tflbm_01_" + str(i).zfill(4) + ".h5"
    f = h5py.File(name, 'r')
    #print(f.keys())
    vel = np.array(f['Velocity_0'][:])
    #charge = np.array(f['Density_0'][:]) + np.array(f['Gamma'][:]) - 1.0
    boundary = np.array(f['Gamma'][:])
    #charge_size = np.sqrt(charge.shape[0])
    nx = f['Nx'][:][0]
    ny = f['Ny'][:][0]
    nz = f['Nz'][:][0]
    #charge = charge.reshape(nx, ny, nz, 3)
    vel = vel.reshape(nx, ny, nz, 3)
    boundary = boundary.reshape(nx, ny, nz, 1)
    #charge = np.concatenate([charge, charge, charge], 3)/10.0
    
    #charge = charge.reshape(charge_size, charge_size, 1)
    #charge = charge.reshape(512, 1024, 3)
    #charge = charge.reshape(512, 512, 3)
    #charge = charge.reshape(1024, 1024, 3)
    #charge = charge.reshape(2048, 2048, 3)
    #charge = np.array(f['Sigma'][:])
    #charge = charge.reshape(200,200,1)
    #div = divergence(charge)
    print(np.max(vel[:,:,:]))
    print(np.min(vel[:,:,:]))
    #charge = div
    #charge = (charge - charge.min()) * 3.0
    #charge = (charge - charge.min())/charge.max()
    #plt.imshow(charge[:,:,0])
    #plt.imshow(charge)
    #plt.imshow(charge[:,:,0])
    #plt.imshow(np.sqrt((charge[:,:,0] * charge[:,:,0])))
    if show=="bounds":
      
      X = []
      Y = []
      Z = []
      for i in xrange(nx-2):
        for j in xrange(ny-2):
          for k in xrange(nz-2):
            if boundary[i+1,j+1,k+1] > 0:
              if not(boundary[i,j+1,k+1] > 0 and boundary[i+1,j,k+1] > 0 and boundary[i+1,j+1,k] > 0 and boundary[i+2,j+1,k+1] > 0 and boundary[i+1,j+2,k+1] > 0 and boundary[i+1,j+1,k+2] > 0):
                X.append(i)
                Y.append(j)
                Z.append(k)
      X = np.array(X)
      Y = np.array(Y)
      Z = np.array(Z)
          
      X_q, Y_q, Z_q = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1), np.arange(nz/2, nz/2+4, 1))
      print(X.shape)
      #colorsMap='jet'
      #cm = plt.get_cmap(colorsMap)
      #cs = vel[:,:,nz/2:nz/2+4,0]
      #cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
      #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(X, Y, Z)
      #ax.scatter(X_q, Y_q, Z_q, c=scalarMap.to_rgba(cs))
      #scalarMap.set_array(cs)
      #ax.scatter(X_q, Y_q, Z_q, vel[:,:,nz/2:nz/2+4,0],vel[:,:,nz/2:nz/2+4,1],vel[:,:,nz/2:nz/2+4,2])
      #plt.imshow(np.sqrt((charge[10,:,:,0] * charge[10,:,:,0])  + (charge[10,:,:,1] * charge[10,:,:,1])))
    if show=="flow":
      z_loc = nz/2
      plt.imshow(np.sqrt((vel[z_loc,:,:,0] * vel[z_loc,:,:,0]) + (vel[z_loc,:,:,1] * vel[z_loc,:,:,1]) + (vel[z_loc,:,:,2] * vel[z_loc,:,:,2])))
    #plt.imshow(div)
    plt.show()



