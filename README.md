# Phy-Net

## Introduction

This repository is a looks at compressing lattice boltzmann physics simulations onto neural networks. This approach relies on learning a compressed representation of simulation while learning the dynamics on this compressed form. This allows us to simulate large systems with low memory and computation. We apply this method to Lattice Boltzmann fluid flow simulations. This work is currently being written up.

## Related Work
Similary works can be found in "[Accelerating Eulerian Fluid Simulation With Convolutional Networks](https://arxiv.org/pdf/1607.03597.pdf)" and "[Convolutional Neural Networks for steady Flow Approximation](https://autodeskresearch.com/publications/convolutional-neural-networks-steady-flow-approximation)".

## Network Details

The network learns a encoding, compression, and decoding peice. The encoding piece learns to compress the state of the physics simulation. The compression piece learns the dynamics of the simulation on this compressed piece. The decoding piece learns to decode the compressed representation. The network is kept all convolutional allowing it to be used and trained and evaluated on any size simulation. This means that once the model is trained on a small simulation (say 256 by 256 grid) it can then attempt to simulate the dynamics of a larger simulation (say 1024 by 1024 grid). We show that the model can still produced accurate results even with evaluating larger simulations then seen during training.

## Lattice Boltzmann Fluid Flow

Using the Mechsys library we generated 2D and 3D fluid simulations to train our model. For the 2D case we simulate a variety of random objects interacting with a steady flow and periodic boundary conditions. The simulation is a 256 by 256 grid. Using the trained model we evaluate on grid size 256 by 256, 512 by 512, and 1024 by 1024. Here are examples of generated simulations (better videos being produced now).

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/SsAWHkcENEI/0.jpg)](https://www.youtube.com/watch?v=SsAWHkcENEI)

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/cm-cc7_Djfg/0.jpg)](https://www.youtube.com/watch?v=cm-cc7_Djfg)

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/2G8-OHjZQto/0.jpg)](https://www.youtube.com/watch?v=2G8-OHjZQto)

We can 


3D simulations are currently being generated and trained on. 

## Conclusion

This project is under active development.




