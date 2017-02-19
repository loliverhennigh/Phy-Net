
# Abstract



We present Net-Phy, a method for compressing both the computation time and memory usage of fluid flow simulations using deep neural networks. Net-Phy employs convolutional autoencoders and residual connections in a fully differentiable scheme to reduce the state size of a simulation and learn the dynamics on this compressed form. The result is a small computationaly effecient network that can be itereated and queired to reproduce a fluid simulation or extract desire measurements such as drag and flux. We apply this method to both 2d an 3d fluid flow simulations computed with the Lattice Boltzmann method. We also show that by training on small scale simulations we can use the learned network to generated larger simulations accuratly.



