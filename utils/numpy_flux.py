
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

  return flux_x, flux_y



