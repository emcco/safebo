import torch
import gpytorch
import numpy as np
import globals
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import pyplot as plt

def sample_RKHS(x):
    """
    Creates an RKHS function with randomly chosen centers and alpha values,
    ensuring that approximately 70% of function values are above the safety threshold.
    
    Args:
        x: Input tensor
        safety_threshold: Safety threshold value (default: 0.0)
    
    Returns:
        Tuple of (function values, RKHS norm)
    """
    # Generate random centers between 0 and 10
    num_centers = 5
    centers = torch.rand(num_centers, 1) * 10
    
    # Initialize kernel
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    # Generate random alpha values
    alpha = (torch.rand(num_centers, 1) * 4 - 2)  # Values between -2 and 2
    
    # Evaluate Gram matrix of centers
    K = kernel(centers, centers).evaluate()
    
    # Compute and apply scaling
    scaling_factor = globals.B / torch.sqrt(alpha.transpose(0,1) @ K @ alpha)
    scaled_alpha = scaling_factor * alpha
    
    # Evaluate function at x
    K_x_centers = kernel(x, centers).evaluate()
    rkhs_function = K_x_centers @ scaled_alpha
    jmin = torch.quantile(rkhs_function, 0.3)
    
    # Compute RKHS norm
    norm = torch.sqrt(scaled_alpha.transpose(0,1) @ K @ scaled_alpha)
    
    return rkhs_function, alpha, centers, jmin, norm

def plot_RKHS(rkhs_function, test_x, jmin):

    jmin_np = np.empty(test_x.size(0))
    for i in range(len(jmin_np)):
        jmin_np[i] = jmin

    # Plot observed points
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.figure(figsize=(10, 6))
    ax.plot(test_x.detach().numpy(), rkhs_function.detach().numpy(), 'g', label='True f, RKHS')
    # Plot line for jmin
    ax.plot(test_x.detach().numpy(), jmin_np, 'r-', label='Threshold')
    # Plot training data as black stars
  
    ax.set_xlabel('Input x', fontsize=12)
    ax.set_ylabel('Output y', fontsize=12)
    ax.set_title('Sampled RKHS Function', fontsize=18, fontweight='bold')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)        # Save the plot
    ax.xaxis.set_major_locator(MultipleLocator(1.0))    # Major ticks every 1 unit
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))    # Minor ticks every 0.2 units
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.show()


test_x = torch.linspace(0, 10, 1000)   

rkhs_function, alpha, centers, jmin, norm = sample_RKHS(test_x)
plot_RKHS(rkhs_function, test_x, jmin)
print(f'alpha: {alpha}, centers: {centers}, jmin: {jmin}, norm: {norm}')

print('Done')

