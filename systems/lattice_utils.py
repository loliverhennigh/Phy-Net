

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

def subtract_lattice(lattice, Weights):
  # this increases percesion before converting to 32 bit
  # should change this to just one op
  for i in xrange(lattice.shape[0]):
    for j in xrange(lattice.shape[1]):
      for k in xrange(lattice.shape[1]):
        lattice[i,j,k,:] = lattice[i,j,k,:] - Weights
  return lattice

def add_lattice(lattice, Weights):
  # this increases percesion before converting to 32 bit
  # should change this to just one op
  for i in xrange(lattice.shape[0]):
    for j in xrange(lattice.shape[1]):
      for k in xrange(lattice.shape[1]):
        lattice[i,j,k,:] = lattice[i,j,k,:] + Weights
  return lattice

def lattice_to_vel(lattice, Lveloc):
  # this increases percesion before converting to 32 bit
  # should change this to just one op
  velocity = np.zeros((lattice.shape[0],lattice.shape[1],lattice.shape[2],3))
  for i in xrange(lattice.shape[0]):
    for j in xrange(lattice.shape[1]):
      for k in xrange(lattice.shape[2]):
        for g in xrange(Lveloc.shape[0]):
          velocity[i,j,k,:] = lattice[i,j,k,g] * Lveloc[g,:]
  return velocity

def lattice_to_rho(lattice, Lveloc):
  # this increases percesion before converting to 32 bit
  # should change this to just one op
  rho = np.zeros((lattice.shape[0],lattice.shape[1],lattice.shape[2]))
  for i in xrange(lattice.shape[0]):
    for j in xrange(lattice.shape[1]):
      for k in xrange(lattice.shape[2]):
        for g in xrange(Lveloc.shape[0]):
          rho[i,j,k] += lattice[i,j,k,g]
  return rho 

def rho_to_psi(rho):
  # fake constants
  return 4.0 * np.exp(-200.0/rho)

def lattice_to_force(lattice, boundary, Lveloc):
  # Momentum exchange method (page 135 Lattice Boltzmann method)
  # calc force
  force = np.zeros((lattice.shape[0],lattice.shape[1],lattice.shape[2],3))
  rho = lattice_to_rho(lattice)
  psi = rho_to_psi(rho)
  for i in range(1, lattice.shape[0]-1, 1):
    for j in range(1, lattice.shape[1]-1, 1):
      for k in range(1, lattice.shape[2]-1, 1):
        psi = psi[i, j,k]
        if boundary[i,j,k] < 1.0:
          for g in range(1, lattice.shape[3]):
            nb_psi = psi[i + Lveloc[g,0], j + Lveloc[g,1], k + Lveloc[g,2]]
            if boundary[i + Lveloc[g,0], j + Lveloc[g,1], k + Lveloc[g,2]] > 0.0:
              nb_psi = 1.0
            #force[i,j,k] += 
            
            
             
            
            

