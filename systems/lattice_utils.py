
import numpy as np

# easy to add more
WEIGHTD2Q9 = np.array([4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.])
WEIGHTD3Q15 = np.array([2./9., 1./9., 1./9., 1./9., 1./9.,  1./9.,  1./9., 1./72., 1./72. , 1./72., 1./72., 1./72., 1./72., 1./72., 1./72.])
OPPOSITED2Q9 = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
OPPOSITED3Q15 = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13])
LVELOCD2Q9 = np.array([ [0,0,0], [1,0,0], [0,1,0], [-1,0,0], [0,-1,0], [1,1,0], [-1,1,0], [-1,-1,0], [1,-1,0] ])
LVELOCD3Q15 = np.array([ [ 0, 0, 0], [ 1, 0, 0], [-1, 0, 0], [ 0, 1, 0], [ 0,-1, 0], [ 0, 0, 1], [ 0, 0,-1], [ 1, 1, 1], [-1,-1,-1], [ 1, 1,-1], [-1,-1, 1], [ 1,-1, 1], [-1, 1,-1], [ 1,-1,-1], [-1, 1, 1] ])

def get_weights(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    return WEIGHTD2Q9 
  elif lattice_size == 15:
    return WEIGHTD3Q15 

def get_lveloc(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    return LVELOCD2Q9 
  elif lattice_size == 15:
    return LVELOCD3Q15 

def get_opposite(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    return OPPOSITED2Q9 
  elif lattice_size == 15:
    return OPPOSITED3Q15 

def subtract_lattice(lattice, Weights):
  # this increases percesion before converting to 32 bit
  Weights = Weights.reshape((1,1,1,Weights.shape[0]))
  lattice = lattice - Weights
  return lattice

def add_lattice(lattice, Weights):
  # get true lattice state from subtracted compressed
  Weights = Weights.reshape((1,1,1,Weights.shape[0]))
  lattice = lattice + Weights
  return lattice

def lattice_to_vel(lattice, Lveloc):
  # get velocity vector field from lattice
  Lveloc = Lveloc.reshape((1,1,1,Lveloc.shape[0],Lveloc.shape[1]))
  print(lattice.shape)
  lattice = lattice.reshape((lattice.shape[0],lattice.shape[1],lattice.shape[2],lattice.shape[3],1))
  velocity = np.sum(lattice * Lveloc, axis=3)
  return velocity

def lattice_to_divergence(lattice, Lveloc):
  # divergence from lattice (finite differnce method)

  velocity = lattice_to_vel(lattice, Lveloc)

  if velocity.shape[2] == 1:
    velocity_x_0 = velocity[0:-2,1:-1,0,1]
    velocity_x_2 = velocity[2:  ,1:-1,0,1]
    velocity_x = velocity_x_0 - velocity_x_2
  
    velocity_y_0 = velocity[1:-1,0:-2,0,0]
    velocity_y_2 = velocity[1:-1,2:  ,0,0]
    velocity_y = velocity_y_0 - velocity_y_2
   
    div = velocity_x + velocity_y
 
  else:
    velocity_x_0 = velocity[0:-2,1:-1,1:-1,2]
    velocity_x_2 = velocity[2:  ,1:-1,1:-1,2]
    velocity_x = velocity_x_0 - velocity_x_2
  
    velocity_y_0 = velocity[1:-1,0:-2,1:-1,1]
    velocity_y_2 = velocity[1:-1,2:  ,1:-1,1]
    velocity_y = velocity_y_0 - velocity_y_2
   
    velocity_z_0 = velocity[1:-1,1:-1,0:-2,0]
    velocity_z_2 = velocity[1:-1,1:-1,2:  ,0]
    velocity_z = velocity_z_0 - velocity_z_2
 
    div = velocity_x + velocity_y + velocity_z
  
  return np.sum(np.abs(div))

def lattice_to_flux(lattice, boundary, Lveloc):
  # flux from lattice
  velocity = lattice_to_vel(lattice, Lveloc)
  rho = lattice_to_rho(lattice)
  flux = velocity * rho * (-boundary + 1.0)
  return flux

def vel_to_norm_vel(velocity):
  # norm of velocity vector field from velocity vector field
  norm_vel = np.sqrt(np.square(velocity[:,:,:,0:1]) + np.square(velocity[:,:,:,1:2]) + np.square(velocity[:,:,:,2:3]))
  return norm_vel

def lattice_to_rho(lattice):
  # lattice to density
  rho = np.sum(lattice, axis=3) 
  rho = rho.reshape((rho.shape[0], rho.shape[1], rho.shape[2], 1)) 
  return rho 

def rho_to_psi(rho):
  # density to psi
  # fake constants (currently unused function)
  return 4.0 * np.exp(-200.0/rho)

def lattice_to_force(lattice, boundary, Lveloc):
  # Momentum exchange method (page 135 Lattice Boltzmann method)
  # calc sum of force on all boundarys
  force = np.zeros((3))
  for i in range(1, lattice.shape[0]-1, 1):
    for j in range(1, lattice.shape[1]-1, 1):
      if lattice.shape[2] == 1:
       for g in range(1, lattice.shape[3]):
         if boundary[i, j, 0, 0] < 1.0 and boundary[i + Lveloc[g,0], j + Lveloc[g,1], 0 + Lveloc[g,2], 0] > 0.0:
          force += Lveloc[g] * lattice[i,j,0,g]
      else:
        for k in range(1, lattice.shape[2]-1, 1):
          for g in range(1, lattice.shape[3]):
            if boundary[i, j, k, 0] < 1.0 and boundary[i + Lveloc[g,0], j + Lveloc[g,1], k + Lveloc[g,2], 0] > 0.0:
                force += Lveloc[g] * lattice[i,j,k,g]
              
  return force

def pad_2d_to_3d(tensor):
  # pad a dim to make 3d
  tensor = tensor.reshape((tensor.shape[0],tensor.shape[1],1,tensor.shape[2]))
  return tensor




 
