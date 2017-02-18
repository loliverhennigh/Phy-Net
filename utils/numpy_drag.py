
import numpy as np

def drag_2d(velocity_field, density_field, boundary):

  force_x = 0.0
  force_y = 0.0

  # unormalize density
  density_field = density_field + 1.0

  for i in xrange(boundary.shape[0]-2): # not on edges
    for j in xrange(boundary.shape[1]-2):
      if boundary[i+1,j+1] < 1.0:
        if boundary[i+2,j+1] > 0.0:
          force_x += .5*density_field[i+1,j+1]*np.sqrt(pow(velocity_field[i+1,j+1,0],2) + pow(velocity_field[i+1,j+1,1],2))

        if boundary[i,j+1] > 0.0:
          #print("-x " + str(pressure_field[i+1,j+1]))
          force_x -= .5*density_field[i+1,j+1]*np.sqrt(pow(velocity_field[i+1,j+1,0],2) + pow(velocity_field[i+1,j+1,1],2))

        if boundary[i+1,j+2] > 0.0:
          #print("+y " + str(pressure_field[i+1,j+1]))
          force_y += .5*density_field[i+1,j+1]*np.sqrt(pow(velocity_field[i+1,j+1,0],2) + pow(velocity_field[i+1,j+1,1],2))

        if boundary[i+1,j] > 0.0:
          #print("-y " + str(pressure_field[i+1,j+1]))
          force_y -= .5*density_field[i+1,j+1]*np.sqrt(pow(velocity_field[i+1,j+1,0],2) + pow(velocity_field[i+1,j+1,1],2))

      
  return force_x, force_y

def drag_3d(velocity_field, density_field, boundary):

  force_x = 0.0
  force_y = 0.0
  force_z = 0.0

  # unormalize density
  density_field = density_field + 1.0

  for i in xrange(boundary.shape[0]-2): # not on edges
    for j in xrange(boundary.shape[1]-2):
      for k in xrange(boundary.shape[2]-2):
        if boundary[i+1,j+1,k+1] < 1.0:
          if boundary[i+2,j+1,k+1] > 0.0:
            force_x += .5*density_field[i+1,j+1,k+1]*np.sqrt(pow(velocity_field[i+1,j+1,k+1,0],2) + pow(velocity_field[i+1,j+1,k+1,1],2) + pow(velocity_field[i+1,j+1,k+1,2],2))
  
          if boundary[i,j+1,k+1] > 0.0:
            force_x -= .5*density_field[i+1,j+1,k+1]*np.sqrt(pow(velocity_field[i+1,j+1,k+1,0],2) + pow(velocity_field[i+1,j+1,k+1,1],2) + pow(velocity_field[i+1,j+1,k+1,2],2))
  
          if boundary[i+1,j+2,k+1] > 0.0:
            force_y += .5*density_field[i+1,j+1,k+1]*np.sqrt(pow(velocity_field[i+1,j+1,k+1,0],2) + pow(velocity_field[i+1,j+1,k+1,1],2) + pow(velocity_field[i+1,j+1,k+1,2],2))
  
          if boundary[i+1,j,k+1] > 0.0:
            force_y -= .5*density_field[i+1,j+1,k+1]*np.sqrt(pow(velocity_field[i+1,j+1,k+1,0],2) + pow(velocity_field[i+1,j+1,k+1,1],2) + pow(velocity_field[i+1,j+1,k+1,2],2))

          if boundary[i+1,j+1,k] > 0.0:
            force_z += .5*density_field[i+1,j+1,k+1]*np.sqrt(pow(velocity_field[i+1,j+1,k+1,0],2) + pow(velocity_field[i+1,j+1,k+1,1],2) + pow(velocity_field[i+1,j+1,k+1,2],2))

          if boundary[i+1,j+1,k+2] > 0.0:
            force_z -= .5*density_field[i+1,j+1,k+1]*np.sqrt(pow(velocity_field[i+1,j+1,k+1,0],2) + pow(velocity_field[i+1,j+1,k+1,1],2) + pow(velocity_field[i+1,j+1,k+1,2],2))
      
  return force_x, force_y, force_z




