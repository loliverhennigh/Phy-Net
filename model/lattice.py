
import tensorflow as tf
import numpy as np
from model.nn import int_shape


# easy to add more
VELOCITY_KERNEL_2D = np.zeros((3,3,2,1))
VELOCITY_KERNEL_2D[2,1,0,0] =  1.0
VELOCITY_KERNEL_2D[0,1,0,0] = -1.0
VELOCITY_KERNEL_2D[1,2,1,0] =  1.0
VELOCITY_KERNEL_2D[1,0,1,0] = -1.0
VELOCITY_KERNEL_3D = np.zeros((3,3,3,3,1))
VELOCITY_KERNEL_3D[2,1,1,2,0] =  1.0
VELOCITY_KERNEL_3D[0,1,1,2,0] = -1.0
VELOCITY_KERNEL_3D[1,2,1,1,0] =  1.0
VELOCITY_KERNEL_3D[1,0,1,1,0] = -1.0
VELOCITY_KERNEL_3D[1,1,2,0,0] =  1.0
VELOCITY_KERNEL_3D[1,1,0,0,0] = -1.0

BOUNDARY_EDGE_KERNEL_2D = np.zeros((3,3,9,1))
BOUNDARY_EDGE_KERNEL_2D[1,0,1,0] = 1.0 # right
BOUNDARY_EDGE_KERNEL_2D[0,1,2,0] = 1.0 # up
BOUNDARY_EDGE_KERNEL_2D[1,2,3,0] = 1.0 # left
BOUNDARY_EDGE_KERNEL_2D[2,1,4,0] = 1.0 # down
BOUNDARY_EDGE_KERNEL_2D[0,0,5,0] = 1.0 # up right
BOUNDARY_EDGE_KERNEL_2D[0,2,6,0] = 1.0 # up left
BOUNDARY_EDGE_KERNEL_2D[2,2,7,0] = 1.0 # down left
BOUNDARY_EDGE_KERNEL_2D[2,0,8,0] = 1.0 # down right



# maybe correct
BOUNDARY_EDGE_KERNEL_3D = np.zeros((3,3,3,15,1))
BOUNDARY_EDGE_KERNEL_3D[1,1,0,1 ,0] = 1.0 # down
BOUNDARY_EDGE_KERNEL_3D[1,1,2,2 ,0] = 1.0 # up
BOUNDARY_EDGE_KERNEL_3D[1,0,1,3 ,0] = 1.0 # down
BOUNDARY_EDGE_KERNEL_3D[1,2,1,4 ,0] = 1.0 # up
BOUNDARY_EDGE_KERNEL_3D[0,1,1,5 ,0] = 1.0 # down
BOUNDARY_EDGE_KERNEL_3D[2,1,1,6 ,0] = 1.0 # up

#BOUNDARY_EDGE_KERNEL_3D[1,1,0,1 ,0] = 1.0 # left
#BOUNDARY_EDGE_KERNEL_3D[1,1,2,2 ,0] = 1.0 # right
#BOUNDARY_EDGE_KERNEL_3D[1,0,1,3 ,0] = 1.0 # left
#BOUNDARY_EDGE_KERNEL_3D[1,2,1,4 ,0] = 1.0 # right
#BOUNDARY_EDGE_KERNEL_3D[0,1,1,5 ,0] = 1.0 # out
#BOUNDARY_EDGE_KERNEL_3D[2,1,1,6 ,0] = 1.0 # in

BOUNDARY_EDGE_KERNEL_3D[0,0,0,7 ,0] = 1.0 # down left out
BOUNDARY_EDGE_KERNEL_3D[2,2,2,8 ,0] = 1.0 # up right in
BOUNDARY_EDGE_KERNEL_3D[2,0,0,9 ,0] = 1.0 # down left in 
BOUNDARY_EDGE_KERNEL_3D[0,2,2,10,0] = 1.0 # up right out
BOUNDARY_EDGE_KERNEL_3D[0,2,0,11,0] = 1.0 # down right out
BOUNDARY_EDGE_KERNEL_3D[2,0,2,12,0] = 1.0 # up left in 
BOUNDARY_EDGE_KERNEL_3D[2,2,0,13,0] = 1.0 # down right in 
BOUNDARY_EDGE_KERNEL_3D[0,0,2,14,0] = 1.0 # up left out

def simple_conv_2d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv2d(x, k, [1, 1, 1, 1], padding='VALID')
  return y

def simple_trans_conv_2d(x, k):
  """A simplified 2D convolution operation"""
  output_shape = tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(k)[2]]) 
  y = tf.nn.conv2d_transpose(x, k, output_shape, [1, 1, 1, 1], padding='SAME')
  y = tf.reshape(y, [int(x.get_shape()[0]), int(x.get_shape()[1]), int(x.get_shape()[2]), int(k.get_shape()[2])])
  return y

def simple_conv_3d(x, k):
  """A simplified 2D convolution operation"""
  y = tf.nn.conv3d(x, k, [1, 1, 1, 1, 1], padding='VALID')
  return y

def simple_trans_conv_3d(x, k):
  """A simplified 2D convolution operation"""
  print(k.get_shape())
  output_shape = tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], tf.shape(k)[3]]) 
  y = tf.nn.conv3d_transpose(x, k, output_shape, [1, 1, 1, 1, 1], padding='SAME')
  y = tf.reshape(y, [int(x.get_shape()[0]), int(x.get_shape()[1]), int(x.get_shape()[2]), int(x.get_shape()[3]), int(k.get_shape()[3])])
  return y

def get_weights(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    return tf.constant(np.array([4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.]), dtype=1)
  elif lattice_size == 15:
    return tf.constant(np.array([2./9., 1./9., 1./9., 1./9., 1./9.,  1./9.,  1./9., 1./72., 1./72. , 1./72., 1./72., 1./72., 1./72., 1./72., 1./72.]), dtype=1)

def get_lveloc(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    return tf.constant(np.array([ [0,0], [0,1], [1,0], [0,-1], [-1,0], [1,1], [1,-1], [-1,-1], [-1,1] ]), dtype=1)
  elif lattice_size == 15:
    #return tf.constant(np.array([ [ 0, 0, 0], [ 1, 0, 0], [-1, 0, 0], [ 0, 1, 0], [ 0,-1, 0], [ 0, 0, 1], [ 0, 0,-1], [ 1, 1, 1], [-1,-1,-1], [ 1, 1,-1], [-1,-1, 1], [ 1,-1, 1], [-1, 1,-1], [ 1,-1,-1], [-1, 1, 1] ]), dtype=1)
    return tf.constant(np.array([ [ 0, 0, 0], [ 1, 0, 0], [-1, 0, 0], [ 0, 1, 0], [ 0,-1, 0], [ 0, 0, 1], [ 0, 0,-1], [ 1, 1, 1], [-1,-1,-1], [ 1, 1,-1], [-1,-1, 1], [ 1,-1, 1], [-1, 1,-1], [ 1,-1,-1], [-1, 1, 1] ]), dtype=1)

def get_lelect():
  D1 = tf.constant(np.array([ [-0.5, 0.5, 0.0], [-0.5,-0.5, 0.0], [ 0.5,-0.5, 0.0], [ 0.5, 0.5, 0.0], [-0.5, 0.0, 0.5], [-0.5, 0.0,-0.5], [ 0.5, 0.0,-0.5], [ 0.5, 0.0, 0.5], [ 0.0,-0.5, 0.5], [ 0.0,-0.5,-0.5], [ 0.0, 0.5,-0.5], [ 0.0, 0.5, 0.5] ]), dtype=1)
  D2 =  tf.constant(np.array([ [ 0.5,-0.5, 0.0], [ 0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5,-0.5, 0.0], [ 0.5, 0.0,-0.5], [ 0.5, 0.0, 0.5], [-0.5, 0.0, 0.5], [-0.5, 0.0,-0.5], [ 0.0, 0.5,-0.5], [ 0.0, 0.5, 0.5], [ 0.0,-0.5, 0.5], [ 0.0,-0.5,-0.5] ]), dtype=1)
  return D1, D2

def get_lmagne():
  H1 = tf.constant(np.array([ [ 0.0, 0.0, 1.0], [ 0.0, 0.0, 1.0], [ 0.0, 0.0, 1.0], [ 0.0, 0.0, 1.0], [ 0.0,-1.0, 0.0], [ 0.0,-1.0, 0.0], [ 0.0,-1.0, 0.0], [ 0.0,-1.0, 0.0], [ 1.0, 0.0, 0.0], [ 1.0, 0.0, 0.0], [ 1.0, 0.0, 0.0], [ 1.0, 0.0, 0.0] ]), dtype=1)
  H2 =  tf.constant(np.array([ [ 0.0, 0.0,-1.0], [ 0.0, 0.0,-1.0], [ 0.0, 0.0,-1.0], [ 0.0, 0.0,-1.0], [ 0.0, 1.0, 0.0], [ 0.0, 1.0, 0.0], [ 0.0, 1.0, 0.0], [ 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0] ]), dtype=1)
  return H1, H2

def get_opposite(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    return tf.constant(np.array([0, 3, 4, 1, 2, 7, 8, 5, 6]), dtype=1)
  elif lattice_size == 15:
    return tf.constant(np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]), dtype=1)

def get_velocity_kernel(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    return tf.constant(VELOCITY_KERNEL_2D, dtype=1)
  elif lattice_size == 15:
    return tf.constant(VELOCITY_KERNEL_3D, dtype=1)

def get_edge_kernel(lattice_size):
  # returns the lattice weights given the size of lattice
  if lattice_size == 9:
    return tf.constant(BOUNDARY_EDGE_KERNEL_2D, dtype=1)
  elif lattice_size == 15:
    return tf.constant(BOUNDARY_EDGE_KERNEL_3D, dtype=1)

def subtract_lattice(lattice):
  Weights = get_weights(int(lattice.get_shape()[-1]))
  dims = len(lattice.get_shape())-1
  Weights = tf.reshape(Weights, dims*[1] + [int(Weights.get_shape()[0])])
  lattice = lattice - Weights
  return lattice

def add_lattice(lattice):
  Weights = get_weights(int(lattice.get_shape()[-1]))
  dims = len(lattice.get_shape())-1
  Weights = tf.reshape(Weights, dims*[1] + [int(Weights.get_shape()[0])])
  lattice = lattice + Weights
  return lattice

def lattice_to_vel(lattice):
  # get velocity vector field from lattice
  Lveloc = get_lveloc(int(lattice.get_shape()[-1]))
  dims = len(lattice.get_shape())-1
  Lveloc_shape = list(map(int, Lveloc.get_shape()))
  Lveloc = tf.reshape(Lveloc, dims*[1] + Lveloc_shape)
  lattice_shape = list(map(int, lattice.get_shape()))
  lattice = tf.reshape(lattice, lattice_shape + [1])
  velocity = tf.reduce_sum(Lveloc * lattice, axis=dims)
  return velocity

def vel_to_norm(velocity):
  if len(velocity.get_shape()) == 4:
    velocity_norm = tf.sqrt(tf.square(velocity[:,:,:,0:1]) + tf.square(velocity[:,:,:,1:2]))
  else:
    velocity_norm = tf.sqrt(tf.square(velocity[:,:,:,:,0:1]) + tf.square(velocity[:,:,:,:,1:2]) + tf.square(velocity[:,:,:,:,2:3]))
  return velocity_norm

def lattice_to_rho(lattice):
  dims = len(lattice.get_shape())-1
  rho = tf.reduce_sum(lattice, axis=dims)
  rho = tf.expand_dims(rho, axis=dims)
  return rho

def lattice_to_divergence(lattice):
  velocity = lattice_to_vel(lattice)
  velocity_shape = list(map(int, velocity.get_shape()))
  velocity_kernel = get_velocity_kernel(int(lattice.get_shape()[-1]))
  if len(velocity_shape) == 4:
    divergence = simple_conv_2d(velocity, velocity_kernel)
    divergence = divergence[:,1:-1,1:-1,:]
  else:
    divergence = simple_conv_3d(velocity, velocity_kernel)
    divergence = divergence[:,1:-1,1:-1,1:-1,:]
  return divergence

def lattice_to_flux(lattice, boundary):
  Lveloc = get_lveloc(int(lattice.get_shape()[-1]))
  rho = lattice_to_rho(lattice)
  velocity = lattice_to_vel(lattice)
  flux = velocity * rho * (-boundary + 1.0)
  return flux

def lattice_to_force(lattice, boundary):
  Lveloc = get_lveloc(int(lattice.get_shape()[-1]))
  dims = len(lattice.get_shape())-1
  Lveloc_shape = list(map(int, Lveloc.get_shape()))
  Lveloc = tf.reshape(Lveloc, dims*[1] + Lveloc_shape)
  boundary_shape = list(map(int, boundary.get_shape()))
  boundary_edge_kernel = get_edge_kernel(int(lattice.get_shape()[-1]))
  if len(boundary.get_shape()) == 4:
    edge = simple_trans_conv_2d(boundary,boundary_edge_kernel) 
    edge = edge[:,1:-1,1:-1,:]
    boundary = boundary[:,1:-1,1:-1,:]
    lattice = lattice[:,1:-1,1:-1,:]
  else: 
    """
    top = boundary[:,-1:]
    bottom = boundary[:,:1]
    boundary = tf.concat([top, boundary, bottom], axis=1)
    left = boundary[:,:,-1:]
    right = boundary[:,:,:1]
    boundary = tf.concat([left, boundary, right], axis=2)
    """
    top = boundary[:,-1:]
    boundary = tf.concat([top, boundary], axis=1)
    left = boundary[:,:,-1:]
    boundary = tf.concat([left, boundary], axis=2)
    edge = simple_trans_conv_3d(boundary, boundary_edge_kernel)
    """
    top = lattice[:,-1:]
    bottom = lattice[:,:1]
    lattice = tf.concat([top, lattice, bottom], axis=1)
    left = lattice[:,:,-1:]
    right = lattice[:,:,:1]
    lattice = tf.concat([left, lattice, right], axis=2)
    """
    top = lattice[:,-1:]
    lattice = tf.concat([top, lattice], axis=1)
    left = lattice[:,:,-1:]
    lattice = tf.concat([left, lattice], axis=2)
  edge = edge * (-boundary + 1.0)
  edge = edge * lattice
  edge_shape = list(map(int, edge.get_shape()))
  edge = tf.reshape(edge, edge_shape + [1])
  force = tf.reduce_sum(edge * Lveloc, axis=dims)
  return force, edge[:,:,:,:,:,0]

def lattice_to_electric(lattice, boundary):
  dims = len(lattice.get_shape())-1
  split_lattice = tf.split(lattice, 48, axis=3)
  e_1 = split_lattice[0::4]
  e_2 = split_lattice[1::4]
  e_1 = tf.stack(e_1, axis=3)
  e_2 = tf.stack(e_2, axis=3)
  D1, D2 = get_lelect()
  D1_shape = list(map(int, D1.get_shape()))
  D1 = tf.reshape(D1, dims*[1] + D1_shape)
  D2 = tf.reshape(D2, dims*[1] + D1_shape)
  electric = tf.reduce_sum((D1 * e_1) + (D2 * e_2), axis=dims)
  electric = electric/boundary
  return electric

def lattice_to_magnetic(lattice):
  dims = len(lattice.get_shape())-1
  split_lattice = tf.split(lattice, 48, axis=3)
  m_1 = split_lattice[2::4]
  m_2 = split_lattice[3::4]
  m_1 = tf.stack(m_1, axis=3)
  m_2 = tf.stack(m_2, axis=3)
  H1, H2 = get_lmagne()
  H1_shape = list(map(int, H1.get_shape()))
  H1 = tf.reshape(H1, dims*[1] + H1_shape)
  H2 = tf.reshape(H2, dims*[1] + H1_shape)
  magnetic = tf.reduce_sum((H1 * m_1) + (H2 * m_2), axis=dims)
  #magnetic = tf.reduce_sum((H2 * m_1), axis=dims)
  #magnetic = m_1[:,:,:,4,:] + m_1[:,:,:,5,:] + m_1[:,:,:,6,:] + m_1[:,:,:,7,:]
  #magnetic = m_1[:,:,:,4,:] + m_1[:,:,:,5,:] + m_1[:,:,:,6,:] + m_1[:,:,:,7,:]
  return magnetic 

def field_to_norm(field):
  field_norm = tf.sqrt(tf.square(field[:,:,:,0:1]) + tf.square(field[:,:,:,1:2]) + tf.square(field[:,:,:,2:3]))
  return field_norm



 


