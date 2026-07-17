TUBITAK-2247-C-Project

1. Project Overview

This repository hosts the computational research conducted under the TÜBİTAK 2247-C STAR Fellowship. The study focuses on the mathematical modeling of complex neural dynamics, specifically investigating how large-scale brain oscillations emerge from the interplay of individual neural population interactions. By utilizing mean-field theory, the project creates a bridge between theoretical neuroscience and high-performance numerical simulation.

2. Research Problem

Neural systems are characterized by non-linear responses and sensitive dependencies on initial conditions. Understanding these systems requires solving high-dimensional systems of Ordinary Differential Equations (ODEs). This project aims to characterize stability transitions, rhythmic oscillations, and chaotic regimes within these neural populations to better understand the mechanisms behind signal processing in the brain.

3. Technical Architecture & Implementation

The software is engineered for maximum performance and research flexibility, utilizing the following stack:

A. Numerical Engine

  **Integration Methods: Custom implementation of solvers designed to handle the non-linearities and potential stiffness inherent in neural models.
  **Precision Management: Systematic validation of integration time steps to minimize cumulative numerical errors.

B. High-Performance Optimization (Numba)

  **Standard Python often lacks the execution speed required for large-scale parameter sweeping. This repository utilizes Numba's Just-In-Time (JIT) compilation to compile Python functions into machine code at runtime.
  **This approach enables massive reductions in simulation execution time, allowing for deeper exploration of parameter space (e.g., thousands of simulations per parameter set).

C. Analytical Framework

  **Bifurcation Analysis: Automated scripts detect and plot bifurcation points to identify the threshold parameters where the system shifts from a stable resting state to periodic or chaotic behavior.
  **Chaos Quantification: Calculation of Lyapunov Exponents provides a quantitative measure of the system's sensitivity to initial conditions, distinguishing between predictable oscillations and true chaotic transitions.

4. Key Features & Tools

    Automated Parameter Sweeping: Easily run batch simulations across multidimensional parameter spaces to identify phase transitions.
    Integrated Visualization Suite:
        Phase Portraits: Real-time generation of phase-space trajectories.
        Bifurcation Diagrams: Clear visualization of stability boundaries.
        Spectral Analysis: Power spectral density estimation to analyze frequency components of neural oscillations.
    Modular API: Designed to allow the integration of new neuronal models or mean-field assumptions without requiring modification of the core solver engine.

How to Contribute / Extend

The project is built to be extensible. To add a new neuronal model:

   1- Define the model's derivative function in a new module.

   2- Pass the function reference to the solver engine.

   3- Adjust parameters in params.yaml and execute the simulation.

Contact Information

Denizhan Alan

Electrical and Electronics Engineering Student | Piri Reis University

dalan147@outlook.com
