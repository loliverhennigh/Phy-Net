

# Related Papers

Here is a list of related papers. This will become the bibliography after a full liturature review. Right now this will just contain the most relevent papers.

## Speeding up Fluid Flow with Neural Networks

There are two recent papers using neural networks to speed up the computation time of fluid simulations

### [Convolutional Neural Networks for Steady Flow Approximation](https://autodeskresearch.com/sites/default/files/ADSK-KDD2016.pdf) (Aug 13, 2016)

This paper uses convolutional networks to predict the laminar flow on different objects with 100x speed increases. They use an interesting Signed Distance Function for the boundary conditions. They use OpenBL for simulations. Great citations throughout paper.

### [Accelerating Eulerian Fluid Simulation With Convolutional Networks](https://github.com/google/FluidNet) (2017 (on going))
This paper looks at using neural networks to accelerate eulerian fluid simulations. The main objective of this method is for animations. This paper is still under fairly active development however contains great citations.

## Lattice Boltzmann Method

### [Multi-GPU performance of incompressible flow computation by lattice Boltzmann method on GPU cluster](http://www.sciencedirect.com/science/article/pii/S0167819111000214)

STILL TRYING TO GET FULL PAPER. This looks like a nice comparison paper from Tokyo. The key result is a D3R15 simulation of grid size 2000x1000x1000 with 100 gpus in 6 hours. The current neural network model should be able to handle this size.

### [Large-scale LES Wind Simulation using Lattice Boltzmann Method](http://www.sim.gsic.titech.ac.jp/TSUBAME_ESJ/ESJ_09E.pdf)

Classic paper on a huge 10,000x10,000x512 grid simulation of Tokyo with 4,096 gpus. Not pure Lattice Boltzmann. Uses corrections with Smagorinsky model. Nice to get a sense of the scale

### [Performance Evaluation of Parallel Large-Scale Lattice Boltzmann Applications on Three Supercomputing Architectures](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.320.5541&rep=rep1&type=pdf)

Interesting look at using LBM code on supercomputers. Funny to think that 1 TFlop was a big deal back then.

## Data Driven Surrogate Models

### [Surrogate-Based Aerodynamic Design Optimization: Use of Surrogates in Aerodynamic Design Optimization](http://www.mtc.edu.eg/asat13/pdf/AE14.pdf)
This paper gives a nice overview of surrogate models and their use in aerodynamics. They only look at small number of design paramaters and mention the curse of dimensionality. Accelerating Eulerian Fluid Simulations Cites this work. They list the main types of Surrogate models to be Polynomial Regression, Kriging (gaussian stochastic process models), Radial Basis Functions, Multiple Adaptive Regression Splines, and Neural Networks. Very good overview of surrogate models.

## Reduced-Order Modeling

This is the technique of reducing the computational complexity of mathematical models in numerical simulations. [Here](https://www.reddit.com/r/CFD/comments/5q2t3s/could_someone_explain_reduced_order_modelling_and/?st=iyz2tlfy&sh=151e76dc) is a very breif look at them for CFD. [The wikipedia](https://en.wikipedia.org/wiki/Model_order_reduction) page on them gives a nice over view of different methods used. There is a ton of work in this area but the main idea is to reduce the dimensionality of the model using some kind of projection.

### [Reduced-order modeling: new approaches for computational physics](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20010018414.pdf) (Dec, 2003)

This paper offers a good over view of what Reduced order modeling is. They look at a time domain version of the Volterra Theory. Volterra Series is a model for non-linear behavior. They use this to model the fluid flow.

#### Good Quotes
The intent in coistructig such reduced order models (ROMs) is twofold: to provide quantitatively accurate descriptions of the dynamics of systems at a computational cost much lower than the original numerical model, and to provide a means by which system dynamcis can be readily interpreted

The reduction in computational cost needed to solve the ROM is offset by a potential loss of accuracy and model robustness

The general purpose of reduced-order modeling is to lower the computational DOFs present in numerical model while retaining the model's fidelity.

Neural networks have also been used to develop nonlinear models of unsteady aerodynamics and nonlinear models of maneuvers (Modelling and Identification of Non-Linear Unsteady Aerodynamics Loads by Neural Networks and Genetic Algorithms)

### [Reduced Order Modeling of Fluid/Structure Interaction](http://www.sandia.gov/~ikalash/rom_ldrd_sand.pdf) (2009)

Very rich source of information on reduced order modeling. The explination of Proper Orthogonal Decomposition (POD) (or PCA for machine learning people) for ROM is comprehensable.

### [Introduction to Model Order Reduction](http://www.springer.com/cda/content/document/cda_downloaddocument/9783540788409-c1.pdf?SGWID=0-0-45-773840-p173830213) (2008)

On reading list

### [Reduced-order modeling Applications: Introduction and preliminaries](http://scala.uc3m.es/essim2013/pdf/1_preliminaries.pdf) (2013)

Presentation on Reduced Order Model applications. These slides gives the impression that the projection techneques are usually PCA although they call them proper orthogonal decomposition (POD). They cite the Reduced-order modeling: new approaches for computational physics paper.

#### Good Quotes

There is a large variety of ROMs in the market. They are also known as surrogate models


