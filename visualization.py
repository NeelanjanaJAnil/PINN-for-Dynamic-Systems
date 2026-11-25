"""
Visualization module for rendering final plots.
"""
import matplotlib.pyplot as plt
import os

def plot_results(t, x_true, x_noisy, x_reg, x_pinn, output_dir="."):
    """
    Generates comparison subplots analyzing the three techniques: 
    1. Exact solution based purely on mechanics
    2. Data-driven regression without physical constraint
    3. PINN integrating data-driven learning and physical equations
    """
    # Create the comparison figure
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Modeling a Spring-Mass System: Analytical vs Regression vs PINN', fontsize=16)
    
    # -------------------------------------------------------------
    # Plot 1: Analytical solution vs Time (Ground Truth)
    # -------------------------------------------------------------
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, x_true, label='Analytical (Ground Truth)', color='black', linewidth=2)
    ax1.scatter(t, x_noisy, label='Noisy Measurements', color='gray', s=15, alpha=0.6, marker='o')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Displacement (x)')
    ax1.set_title('Analytical Solution & Observed Data')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # -------------------------------------------------------------
    # Plot 2: Regression formulation vs Time
    # -------------------------------------------------------------
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, x_true, label='Ground Truth', color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.plot(t, x_reg, label='Polynomial Regression', color='blue', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Displacement (x)')
    ax2.set_title('Data-Driven Model (Regression)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # -------------------------------------------------------------
    # Plot 3: PINN prediction vs Time
    # -------------------------------------------------------------
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(t, x_true, label='Ground Truth', color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.plot(t, x_pinn, label='PINN Prediction', color='red', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Displacement (x)')
    ax3.set_title('Physics-Informed Neural Network')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # -------------------------------------------------------------
    # Plot 4: Final comparison
    # -------------------------------------------------------------
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(t, x_true, label='Ground Truth', color='black', linewidth=2)
    ax4.plot(t, x_reg, label='Regression', color='blue', linestyle='-.', linewidth=2, alpha=0.8)
    ax4.plot(t, x_pinn, label='PINN', color='red', linestyle='--', linewidth=2, alpha=0.9)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Displacement (x)')
    ax4.set_title('Final Comparison')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save output representation
    save_path = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved successfully to: {save_path}")
    
    # Display the plot
    plt.show()

def plot_loss_history(loss_history, data_loss_history, physics_loss_history, output_dir="."):
    """
    Plots the training loss history of the PINN model over epochs.
    """
    fig = plt.figure(figsize=(10, 6))
    epochs = range(1, len(loss_history) + 1)
    
    plt.plot(epochs, loss_history, label='Total Loss', color='black', linewidth=2)
    plt.plot(epochs, data_loss_history, label='Data Loss', color='blue', linestyle='--', linewidth=1.5)
    plt.plot(epochs, physics_loss_history, label='Physics Loss', color='red', linestyle='-.', linewidth=1.5)
    
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.title('PINN Training Loss History')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save output representation
    save_path = os.path.join(output_dir, 'loss_history.png')
    plt.savefig(save_path, dpi=300)
    print(f"Loss history plot saved successfully to: {save_path}")
    
    # Display the plot
    plt.show()
