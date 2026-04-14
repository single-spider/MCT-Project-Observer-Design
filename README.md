
# Planar Quadrotor: Observer Design and Output-Feedback Control

**Course:** ES 613: Modern Control Theory (Milestone 4)  
**Authors:** Gaurav Srivastava (23110113) & Mamta Bhambhani (22110045)  
**Institution:** IIT Gandhinagar  

## Overview
This repository contains the mathematical design and Python simulation engine for a Linear Quadratic Estimator (LQE / Kalman Filter) applied to a planar quadrotor. 

In practical flight control, unmeasured velocity states (horizontal velocity, vertical velocity, and pitch rate) must be estimated from available noisy sensor data (GPS, barometer, and IMU measurements). This project synthesizes an optimal Luenberger observer via the dual algebraic Riccati equation to reconstruct the full state vector. The code implements a custom fixed-step stochastic Runge-Kutta 4 (RK4) engine to accurately simulate the combined observer-controller system and validate the Separation Principle.

## Dependencies
The simulation requires Python 3.x and the following standard scientific libraries:
* `numpy`
* `scipy` (Specifically `scipy.linalg.solve_continuous_are` for Riccati equations)
* `matplotlib` (Configured with `seaborn-v0_8-paper` for IEEE-style plotting)

You can install the required packages using:
```bash
pip install numpy scipy matplotlib
````

## Running the Simulation

Execute the main simulation script from the terminal. The engine will solve the LQR and LQE Riccati equations, run the stochastic numerical integration at 1000 Hz (`dt = 0.001`), print a detailed performance summary, and generate the figures.

```bash
python code.py
```

## Simulation Scenarios and Results

The simulation validates the observer performance across four distinct scenarios.

### Scenario A: Observer Convergence

The quadrotor initializes at a perturbed state while the observer initializes with zero knowledge (complete ignorance). The high observer gains drive the estimation errors to zero rapidly, with the slowest velocity modes settling in approximately 0.94 seconds.

### Scenario B: Full-State Feedback vs. Output Feedback

This scenario quantifies the performance degradation introduced by relying on state estimates rather than perfect state feedback. While the initial transient creates a "peaking phenomenon" that temporarily demands higher control effort, the observer-based system successfully stabilizes the quadrotor, proving the Separation Principle.

### Scenario C: Noise Rejection Under Realistic Sensor Uncertainty

We inject Gaussian measurement noise ($v(t) \sim \mathcal{N}(0, 0.01)$) into the output channels while commanding a step response. The Kalman Filter effectively attenuates the high-frequency sensor noise, reducing the measurement variance by approximately 60% while maintaining tight spatial tracking.

### Scenario D: Robustness to Incorrect Initialization

To verify the global asymptotic stability of the error dynamics, the observer is initialized with a state estimate that is $100\%$ opposite to the true physical state. The system successfully recovers from this severe discrepancy and converges within 2.5 seconds.

## Repository Structure

  * `quadrotor_observer.py`: The main simulation script containing the system matrices, Riccati solvers, RK4 engine, and plotting routines.
  * `figures_part4/`: Directory containing the auto-generated high-resolution plots.


## Acknowledgments

The control theory concepts and dual Riccati equation formulations applied in this project are based on standard modern control literature, including *Optimal Control* by Lewis et al. and *Modern Control Engineering* by Ogata.

```
```
