
import numpy as np

def flux_2d(velocity_field, density_field, boundary):

  flux_x = 0.0
  flux_y = 0.0
  density_field = density_field+1.0

  for i in xrange(boundary.shape[0]): # not on edges
    for j in xrange(boundary.shape[1]):
      if boundary[i,j] < 1.0:
        flux_x += velocity_field[i,j,1] * density_field[i,j]
        flux_y += velocity_field[i,j,0] * density_field[i,j]

  flux_x = flux_x / np.sum(1-boundary)
  flux_y = flux_y / np.sum(1-boundary)

  return flux_x, flux_y

def flux_3d(velocity_field, density_field, boundary):

  flux_x = 0.0
  flux_y = 0.0
  flux_z = 0.0
  density_field = density_field+1.0

  for i in xrange(boundary.shape[0]): # not on edges
    for j in xrange(boundary.shape[1]):
      for k in xrange(boundary.shape[2]):
        if boundary[i,j,k] < 1.0:
          flux_x += velocity_field[i,j,k,2] * density_field[i,j,k]
          flux_y += velocity_field[i,j,k,1] * density_field[i,j,k]
          flux_z += velocity_field[i,j,k,0] * density_field[i,j,k]

  flux_x = flux_x / np.sum(1-boundary)
  flux_y = flux_y / np.sum(1-boundary)
  flux_z = flux_z / np.sum(1-boundary)

  return flux_x, flux_y, flux_z




