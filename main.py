"""
Main execution script for the Physics-Informed Neural Network Demonstration Project.
"""
import os
from utils import set_seed
from data_generation import generate_data
from regression_model import train_regression_model, predict_regression
from pinn_model import train_pinn, predict_pinn
from visualization import plot_results, plot_loss_history

def main():
    print("=================================================================")
    print(" Mathematical Modeling & Data-Driven Analysis of Mechanical Systems")
    print(" Using Physics-Informed Neural Networks (PINNs) ")
    print("=================================================================\n")
    
    # 1. Environment Initialization
    set_seed(42)  # Ensures reproducibility across runs
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Defined System Specifications
    m = 1.0     # Mass
    c = 0.5     # Damping coefficient
    k = 4.0     # Stiffness constant
    x0 = 1.0    # Initial displacement x(0)
    v0 = 0.0    # Initial velocity dx/dt(0)

    print("System Physical Parameters:")
    print(f" -> Mass (m)      = {m}")
    print(f" -> Damping (c)   = {c}")
    print(f" -> Stiffness (k) = {k}")
    print(f" -> Initial State = x(0): {x0}, v(0): {v0}")
    print("-----------------------------------------------------------------")
    
    # 2. Derive Analytical Expression & Generate Data
    print("\n[STEP 1 & 2] Evaluating Analytical Solution & Generating Synthetic Target Data...")
    # Time domain: t ∈ [0,10], num_points = 200
    t, x_noisy, x_true = generate_data(
        t_min=0.0, t_max=10.0, num_points=200, 
        m=m, c=c, k=k, x0=x0, v0=v0, noise_std=0.05
    )
    print(" -> Data successfully generated.")
    
    # 3. Standard Regression Data-Driven Model
    print("\n[STEP 3] Training Polynomial Regression (Data-Driven purely)...")
    # Using polynomial regression (degree 6)
    reg_model, poly_features = train_regression_model(t, x_noisy, degree=6)
    x_reg_pred = predict_regression(reg_model, poly_features, t)
    print(" -> Regression model trained and predictions calculated.")
    
    # 4. Physics-Informed Neural Network Model
    print("\n[STEP 4] Training Physics-Informed Neural Network (PINN)...")
    # The PINN learns from both data MSE & physical residuals
    pinn_model, loss_history, data_loss_history, physics_loss_history = train_pinn(
        t, x_noisy, m=m, c=c, k=k, 
        epochs=4000, lr=0.001
    )
    x_pinn_pred = predict_pinn(pinn_model, t)
    print(" -> PINN model training completed.")
    
    # 5. Result Visualization
    print("\n[STEP 5] Rendering analysis visualizations...")
    plot_loss_history(loss_history, data_loss_history, physics_loss_history, output_dir=project_dir)
    plot_results(t, x_true, x_noisy, x_reg_pred, x_pinn_pred, output_dir=project_dir)
    
    print("\n=================================================================")
    print(" Execution Completed. ")
    print(" Please refer to 'loss_history.png' and 'comparison_plot.png' for visualizations.")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
