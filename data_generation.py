"""
Data generation module.
"""
import numpy as np
from analytical_solution import analytical_solution

def generate_data(t_min=0.0, t_max=10.0, num_points=200, m=1.0, c=0.5, k=4.0, x0=1.0, v0=0.0, noise_std=0.05, seed=42):
    """
    Generates synthetic displacement-time data using the analytical solution.
    Adds Gaussian noise to simulate realistic measurement error.
    
    Parameters:
    t_min (float): Start time
    t_max (float): End time
    num_points (int): Number of time points
    m, k (float): Physical system parameters
    x0, v0 (float): Initial conditions
    noise_std (float): Standard deviation of Gaussian noise
    seed (int): Random seed for reproducibility
    
    Returns:
    tuple: (t, x_noisy, x_true) containing time array, noisy dataset, and exact analytical data
    """
    np.random.seed(seed)
    
    # Generate time points
    t = np.linspace(t_min, t_max, num_points)
    
    # True theoretical values
    x_true = analytical_solution(t, m, c, k, x0, v0)
    
    # Generate Gaussian measurement noise
    noise = np.random.normal(loc=0.0, scale=noise_std, size=x_true.shape)
    
    # Noisy observed data
    x_noisy = x_true + noise
    
    return t, x_noisy, x_true
