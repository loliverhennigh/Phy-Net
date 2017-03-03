# Phy-Net

## Introduction

This repository is a look at compressing lattice Boltzmann physics simulations onto neural networks. This approach relies on learning a compressed representation of simulation while learning the dynamics on this compressed form. This allows us to simulate large systems with low memory and computation. We apply this method to Lattice Boltzmann fluid flow simulations. This work is currently being written up.

## Related Work
Similar works can be found in "[Accelerating Eulerian Fluid Simulation With Convolutional Networks](https://arxiv.org/pdf/1607.03597.pdf)" and "[Convolutional Neural Networks for steady Flow Approximation](https://autodeskresearch.com/publications/convolutional-neural-networks-steady-flow-approximation)".

## Network Details

The network learns a encoding, compression, and decoding piece. The encoding piece learns to compress the state of the physics simulation. The compression piece learns the dynamics of the simulation on this compressed piece. The decoding piece learns to decode the compressed representation. The network is kept all convolutional allowing it to be trained and evaluated on any size simulation. This means that once the model is trained on a small simulation (say 256 by 256 grid) it can then attempt to simulate the dynamics of a larger simulation (say 1024 by 1024 grid). We show that the model can still produced accurate results even with larger simulations then seen during training.

## Lattice Boltzmann Fluid Flow

Using the Mechsys library we generated 2D and 3D fluid simulations to train our model. For the 2D case we simulate a variety of random objects interacting with a steady flow and periodic boundary conditions. The simulation is a 256 by 256 grid. Using the trained model we evaluate on grid size 256 by 256, 512 by 512, and 1024 by 1024. Here are examples of generated simulations (network design is being iterated on).

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/tXEqXBAOnws/0.jpg)](https://www.youtube.com/watch?v=tXEqXBAOnws)

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/HUkx8RoxaBw/0.jpg)](https://www.youtube.com/watch?v=HUkx8RoxaBw)

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/1TnNSnyRVmI/0.jpg)](https://www.youtube.com/watch?v=1TnNSnyRVmI)

We can look at various properties of the true versus generated simulations such as mean squared error, divergence of the velocity vector field, drag, and flux. Averaging over several test simulations we see the that the generated simulation produces realistic values. The following graphs show this for 256, 512, and 1024 sized simulations.

![alt tag](https://github.com/loliverhennigh/Phy-Net/blob/master/test/figs/256x256_2d_error_plot.png)
![alt tag](https://github.com/loliverhennigh/Phy-Net/blob/master/test/figs/512x512_2d_error_plot.png)
![alt tag](https://github.com/loliverhennigh/Phy-Net/blob/master/test/figs/1024x1024_2d_error_plot.png)

We notice that the model produces lower then expected y flux for both the 512 and 1024 simulations. This is understandable because the larger simulations tend to have higher y flows then smaller simulations due to the distribution of objects being more clumped. It appears that this effect can be mitigated by changing the object density and distribution (under investigation).

A few more snap shots of simulations

![alt tag](https://github.com/loliverhennigh/Phy-Net/blob/master/test/figs/256x256_2d_flow_image.png)
![alt tag](https://github.com/loliverhennigh/Phy-Net/blob/master/test/figs/512x512_2d_flow_image.png)
![alt tag](https://github.com/loliverhennigh/Phy-Net/blob/master/test/figs/1024x1024_2d_flow_image.png)

3D simulations are currently being generated and trained on. This is the current best model (not trained to convergence).

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/Byhvre_lDzI/0.jpg)](https://www.youtube.com/watch?v=Byhvre_lDzI)

## Conclusion

This project is under active development.




