# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 11:20:57 2025

@author: dell
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### PINN Definition (unchanged) ###
class PINN(nn.Module):
    def __init__(self, num_layers=10, num_neurons=100):
        super(PINN, self).__init__()
        layers = []
        layers.append(nn.Linear(2, num_neurons))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_neurons, num_neurons))
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(num_neurons, 2)  # Output (u, v)
        self.activation = nn.Tanh()
        # Initialize weights using Xavier initialization
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
        nn.init.xavier_normal_(self.output_layer.weight)
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

### Modified Mesh Generation for Multiple Cylinders ###
def generate_cylinders_mesh(cylinders, domain_size=4.0, resolution=100, num_surface_points=200):
    # Grid points
    x = np.linspace(-domain_size/2, domain_size/2, resolution)
    y = np.linspace(-domain_size/2, domain_size/2, resolution)
    X, Y = np.meshgrid(x, y)
    X_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device)
    Y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device)
    coords = torch.stack([X_flat, Y_flat], dim=1)
    
    # Interior mask (exclude points inside any cylinder)
    interior_mask = torch.ones_like(X_flat, dtype=bool)
    for cyl in cylinders:
        cx, cy = cyl['center']
        r = cyl['radius']
        dist = torch.sqrt((X_flat - cx)**2 + (Y_flat - cy)**2)
        interior_mask &= (dist > r)
    
    # Boundary mask (domain boundaries)
    boundary_mask = (torch.abs(X_flat) >= domain_size/2 - 1e-3) | (torch.abs(Y_flat) >= domain_size/2 - 1e-3)
    
    # Generate surface points for all cylinders
    surface_points = []
    surface_centers = []
    for cyl in cylinders:
        cx, cy = cyl['center']
        r = cyl['radius']
        theta = np.linspace(0, 2*np.pi, num_surface_points)
        x_surface = cx + r * np.cos(theta)
        y_surface = cy + r * np.sin(theta)
        points = np.stack([x_surface, y_surface], axis=1)
        centers = np.tile([cx, cy], (num_surface_points, 1))
        surface_points.append(points)
        surface_centers.append(centers)
    
    surface_points = np.concatenate(surface_points, axis=0)
    surface_centers = np.concatenate(surface_centers, axis=0)
    surface_points = torch.tensor(surface_points, dtype=torch.float32, device=device)
    surface_centers = torch.tensor(surface_centers, dtype=torch.float32, device=device)
    
    return {
        'coords': coords, 'X': X, 'Y': Y,
        'boundary_points': coords[boundary_mask],
        'surface_points': surface_points,
        'surface_centers': surface_centers,
        'interior_points': coords[interior_mask],
        'cylinders': cylinders
    }

### Exact Solution (unchanged for far-field) ###
def exact_cylinder_flow(coords, U=1.0, D=1.0):
    x, y = coords[:, 0], coords[:, 1]
    return torch.stack([torch.ones_like(x)*U, torch.zeros_like(y)], dim=1)  # Uniform flow at far-field

### PDE Residual Computation (unchanged) ###
def compute_pde_residual(model, coords):
    coords.requires_grad_(True)
    u_pred = model(coords)
    u, v = u_pred[:, 0], u_pred[:, 1]
    
    # Compute gradients
    u_x = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0][:, 0]
    u_y = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0][:, 1]
    v_x = torch.autograd.grad(v, coords, torch.ones_like(v), create_graph=True)[0][:, 0]
    v_y = torch.autograd.grad(v, coords, torch.ones_like(v), create_graph=True)[0][:, 1]
    
    continuity = u_x + v_y
    irrotationality = v_x - u_y
    
    return continuity, irrotationality

### Modified Training Function for Multiple Cylinders ###
def train_pinn(model, mesh, num_epochs=40000, learning_rate=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
    
    boundary_points = mesh['boundary_points']
    surface_points = mesh['surface_points']
    surface_centers = mesh['surface_centers']
    interior_points = mesh['interior_points']
    
    exact_velocity_boundary = exact_cylinder_flow(boundary_points)
    
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Boundary loss (far-field)
        u_pred_boundary = model(boundary_points)
        boundary_loss = torch.mean((u_pred_boundary - exact_velocity_boundary) ** 2)
        
        # Surface loss (no-penetration)
        u_pred_surface = model(surface_points)
        x_surface = surface_points[:, 0]
        y_surface = surface_points[:, 1]
        cx = surface_centers[:, 0]
        cy = surface_centers[:, 1]
        radial_velocity = (u_pred_surface[:, 0] * (x_surface - cx) + 
                          u_pred_surface[:, 1] * (y_surface - cy))
        surface_loss = torch.mean(radial_velocity ** 2)
        
        # PDE loss (interior)
        continuity, irrotationality = compute_pde_residual(model, interior_points)
        pde_loss = torch.mean(continuity**2 + irrotationality**2)
        
        total_loss = boundary_loss + surface_loss + pde_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
        
        losses.append(total_loss.item())
        if (epoch+1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4e}')
    
    return losses

### Modified Visualization for Multiple Cylinders ###
def plot_streamlines(model, mesh):
    with torch.no_grad():
        coords = mesh['coords']
        X, Y = mesh['X'], mesh['Y']
        cylinders = mesh['cylinders']
        
        # Create cylinder mask
        x_coords = coords[:, 0].cpu().numpy()
        y_coords = coords[:, 1].cpu().numpy()
        cylinder_mask = np.zeros_like(x_coords, dtype=bool)
        for cyl in cylinders:
            cx, cy = cyl['center']
            r = cyl['radius']
            dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            cylinder_mask |= (dist < r)
        cylinder_mask = cylinder_mask.reshape(X.shape)
        
        u_pred = model(coords)
        u = u_pred[:, 0].cpu().numpy().reshape(X.shape)
        v = u_pred[:, 1].cpu().numpy().reshape(X.shape)
        vel_mag = np.sqrt(u**2 + v**2)
        
        u[cylinder_mask] = np.nan
        v[cylinder_mask] = np.nan
        vel_mag[cylinder_mask] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    strm = ax.streamplot(X, Y, u, v, color=vel_mag, cmap='viridis', density=2)
    # Draw all cylinders
    for cyl in cylinders:
        cx, cy = cyl['center']
        r = cyl['radius']
        ax.add_patch(plt.Circle((cx, cy), r, color='red', fill=True, alpha=0.6))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("PINN-Predicted Streamlines for Multiple Cylinders")
    plt.xlim(-domain_size/2, domain_size/2)
    plt.ylim(-domain_size/2, domain_size/2)
    cbar = plt.colorbar(strm.lines)
    cbar.set_label("Velocity Magnitude")
    plt.show()

### Main Execution ###
if __name__ == "__main__":
    # Define cylinder parameters
    cylinders = [
        {'center': (0.0, -2.0), 'radius': 1.0},
        {'center': (2.0, 0.0), 'radius': 0.75},
        {'center': (-2.0, 2.0), 'radius': 0.5}
    
        
    ]
    domain_size = 10
    resolution = 200
    num_epochs = 40000
    learning_rate = 0.0005
    
    mesh = generate_cylinders_mesh(cylinders, domain_size, resolution)
    model = PINN().to(device)
    losses = train_pinn(model, mesh, num_epochs, learning_rate)
    
    plt.figure()
    plt.plot(np.arange(1, len(losses)+1), losses)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.show()
    
    plot_streamlines(model, mesh)