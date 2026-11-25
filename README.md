# PINN-for-Dynamic-Systems
A PyTorch Physics-Informed Neural Network (PINN) simulating a damped spring-mass oscillator ( 𝑚 𝑥 ¨ + 𝑐 𝑥 ˙ + 𝑘 𝑥 = 0 m x ¨ +c x ˙ +kx=0). Recovers true dynamic trajectories from noisy data ( 𝜎 = 0.05 σ=0.05) where degree-6 polynomial regression fails.
# Physics-Informed Neural Network (PINN) for a Damped Mechanical Oscillator

A comparative study between pure data-driven modeling and physics-informed deep learning applied to a second-order damped harmonic oscillator ($m\ddot{x} + c\dot{x} + kx = 0$). 

While standard data-driven models (such as degree-6 polynomial regression) overfit high-frequency measurement noise, the Physics-Informed Neural Network (PINN) leverages PyTorch automatic differentiation to enforce the governing ODE residual and initial conditions directly within the loss function. This allows the model to act as a physically-consistent filter, faithfully reconstructing the true continuous-time trajectory.

### Key Technical Specifications
- **Governing Equation:** $m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = 0$ ($m=1.0, c=0.5, k=4.0$) with underdamped response.
- **Initial Conditions:** $x(0) = 1.0\text{ m}, \ \dot{x}(0) = 0.0\text{ m/s}$.
- **Dataset:** 200 synthetic time points ($t \in [0, 10]$) with Gaussian sensor noise ($\sigma = 0.05$).
- **Architecture:** 3-layer MLP (20 neurons/layer) with $\tanh$ activations for smooth second-order derivatives.
- **Loss Formulation:** Multi-objective composite loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + 0.1 \cdot \mathcal{L}_{\text{physics}} + 1.0 \cdot \mathcal{L}_{\text{IC}}$$
- **Optimization:** Adam optimizer ($4000$ epochs).
- **Benchmark:** Evaluated against an exact analytical solution and a scikit-learn degree-6 polynomial regression baseline.
