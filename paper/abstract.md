
# Abstract

We present Net-Phy, a method for compressing both the computation time and size of large scale Physics simulations using neural networks. Net-Phy employs convolutional autoencoders and residual layers in a fully differentiable scheme to reduce the state size of a simulation and learn the dynamics on this compressed form. The result is a small computationaly effecient network that can be itereated and queired to reproduce the desired simulation or extract desire measurements. We apply this method to both Fluid flow and Electormagnetic simulations computed with the Lattice Boltzmann method. Our key result is a network that can simulate large scale fluid flow with only a single GPU and in a fraction of the time. The equivolent simulation would require considurable resources and as minimume of 16 similar GPUs.



