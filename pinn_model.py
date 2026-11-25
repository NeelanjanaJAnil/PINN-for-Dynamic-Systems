"""
Physics-Informed Neural Network (PINN) model module via PyTorch.
"""
import torch
import torch.nn as torch_nn
import torch.optim as optim

class PINN(torch_nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Architecture strictly following requirements:
        # Input: time (1 input feature)
        # Hidden: 3 layers of 20 neurons each, Tanh activation
        # Output: predicted displacement x(t) (1 output feature)
        self.net = torch_nn.Sequential(
            torch_nn.Linear(1, 20),
            torch_nn.Tanh(),
            torch_nn.Linear(20, 20),
            torch_nn.Tanh(),
            torch_nn.Linear(20, 20),
            torch_nn.Tanh(),
            torch_nn.Linear(20, 1)
        )
        
    def forward(self, t):
        return self.net(t)

def train_pinn(t_train, x_train, m=1.0, c=0.5, k=4.0, epochs=4000, lr=0.0005):
    """
    Trains a Physics-Informed Neural Network predicting x(t).
    The loss function combines standard data regression loss (MSE) 
    and a physics-informed loss representing the residual of the differential equation:
    m * d²x/dt² + c * dx/dt + k * x = 0
    """
    model = PINN()
    # [MODIFIED] Set learning rate to 0.0005
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert training data arrays into PyTorch tensors
    # Requires grad on inputs to compute dx/dt and d²x/dt² w.r.t inputs during physics loss calculation
    t_tensor = torch.tensor(t_train, dtype=torch.float32).view(-1, 1).requires_grad_(True)
    x_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
    
    criterion = torch_nn.MSELoss()
    
    loss_history = []
    data_loss_history = []
    physics_loss_history = []
    
    print("\nStarting PINN Training...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward Pass
        # [MODIFIED] Use raw time values directly (no normalization)
        x_pred = model(t_tensor)
        
        # 2. Compute Data Loss (MSE between prediction and noisy measurement)
        data_loss = criterion(x_pred, x_tensor)
        
        # 3. Compute Physics Loss
        # Compute first derivative (dx/dt)
        dx_dt = torch.autograd.grad(
            x_pred, t_tensor, 
            grad_outputs=torch.ones_like(x_pred),
            create_graph=True
        )[0]
        
        # Compute second derivative (d²x/dt²)
        d2x_dt2 = torch.autograd.grad(
            dx_dt, t_tensor,
            grad_outputs=torch.ones_like(dx_dt),
            create_graph=True
        )[0]
        
        # The physics residual R(t) = m * x'' + c * x' + k * x 
        residual = m * d2x_dt2 + c * dx_dt + k * x_pred
        physics_loss = criterion(residual, torch.zeros_like(residual))
        
        # 4. Compute Initial Conditions (IC) Loss
        # [NEW] Enforce explicitly x(0) = 1, x'(0) = 0
        t0 = torch.tensor([[0.0]], requires_grad=True)
        x0_pred = model(t0)
        
        dx0 = torch.autograd.grad(
            x0_pred, t0,
            grad_outputs=torch.ones_like(x0_pred),
            create_graph=True
        )[0]
        
        loss_ic = (x0_pred - 1.0)**2 + (dx0 - 0.0)**2
        # Use mean to convert the tensor to scalar if necessary, though shape is (1,1)
        loss_ic = loss_ic.mean()
        
        # 5. Total Loss
        # [MODIFIED] Added weighting factors (lambda_phy = 0.1, lambda_ic = 1.0)
        lambda_phy = 0.1
        lambda_ic = 1.0
        loss = data_loss + lambda_phy * physics_loss + lambda_ic * loss_ic
        
        loss_history.append(loss.item())
        data_loss_history.append(data_loss.item())
        physics_loss_history.append(physics_loss.item())
        
        # 6. Backpropagation
        loss.backward()
        optimizer.step()
        
        # 7. Logging
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | Data: {data_loss.item():.6f} | Phys: {physics_loss.item():.6f} | IC: {loss_ic.item():.6f} | Total: {loss.item():.6f}")
            
    return model, loss_history, data_loss_history, physics_loss_history

def predict_pinn(model, t_test):
    """
    Predicts displacement using the trained PINN.
    """
    model.eval()
    t_tensor = torch.tensor(t_test, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        x_pred = model(t_tensor).numpy().flatten()
    return x_pred
