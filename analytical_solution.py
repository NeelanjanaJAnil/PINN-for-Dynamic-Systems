"""
Analytical solution module.
Models a spring-mass system governed by a second-order differential equation:
m * d²x/dt² + k * x = 0
"""
import numpy as np

def analytical_solution(t, m=1.0, c=0.5, k=4.0, x0=1.0, v0=0.0):
    """
    Computes the exact analytical solution for the damped spring-mass system.
    Equation: m * d²x/dt² + c * dx/dt + k * x = 0
    
    Parameters:
    t  (numpy.ndarray): Time values
    m  (float): Mass of the system
    c  (float): Damping coefficient
    k  (float): Stiffness constant
    x0 (float): Initial displacement (x(0))
    v0 (float): Initial velocity (dx/dt(0))
    
    Returns:
    numpy.ndarray: Displacement curve over time
    """
    gamma = c / (2 * m)
    discriminant = c**2 - 4 * m * k
    
    if discriminant < 0:
        # Underdamped system
        omega_d = np.sqrt(-discriminant) / (2 * m)
        A = x0
        B = (v0 + gamma * x0) / omega_d
        x = np.exp(-gamma * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    elif discriminant == 0:
        # Critically damped system
        A = x0
        B = v0 + gamma * x0
        x = np.exp(-gamma * t) * (A + B * t)
    else:
        # Overdamped system
        r1 = (-c + np.sqrt(discriminant)) / (2 * m)
        r2 = (-c - np.sqrt(discriminant)) / (2 * m)
        A = (v0 - r2 * x0) / (r1 - r2)
        B = (r1 * x0 - v0) / (r1 - r2)
        x = A * np.exp(r1 * t) + B * np.exp(r2 * t)
        
    return x
