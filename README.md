# eliashberg_MD

A python package that allows one to numerically solve the Eliashberg equations. 
Given an momentum and frequency-dependent interaction potential and the density of states, one can currently calculate:

- T_c (using the eigenvalue approach, or the implicit renormalization method)

- The mass renormalization function Z

- The gap function at T_c (using the eigenvector approach, or the implicit renormalization method)

Planned future capabilities include:

- Including chi, not just Z, in the Eliashberg equations

- Including the particle number equation to know what the density is for a given mu

- Solving the nonlinear Eliashberg equations

- Calculating thermodynamic properties such as specific heat

- Calculating the tunneling DOS

- Analytically Continuing to real frequencies using the Pade approximation or solving the integral equations using the method from Marsiglio

- Parallelizing the code using MPI to do large-scale calculations on a supercomputing cluster

- Going beyond s-wave components of the gap

- All of the above, but for lattice systems, including multi-band systems
