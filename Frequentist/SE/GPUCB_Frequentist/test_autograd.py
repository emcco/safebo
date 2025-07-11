import torch
import gpytorch
import globals


# Define kernel and training data
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
kernel.base_kernel.lengthscale = globals.lengthscale
kernel.outputscale = globals.outputscale


train_x = torch.linspace(0, 10, 5).view(-1, 1).requires_grad_(True)  # 5 points in [0, 10]
train_x_leaf = [torch.tensor(entry, requires_grad=True) for entry in train_x]

# Evaluate kernel matrix

# Compute partial derivatives
N = train_x.size(0)  # Number of training points
gradients = torch.zeros((N, N, train_x.size(1)))  # Store gradients for each pair

for i in range(N):
    for j in range(N):
        # Compute kernel value k(train_x[i], train_x[j])
        k_ij = kernel(train_x_leaf[i], train_x_leaf[j]).evaluate()  # Evaluate single kernel value
        
        k_ij.backward()

        # Store gradient if not None
        if train_x_leaf[i].grad is not None:
            gradients[i, j] = train_x_leaf[i].grad

# Print the gradients
print("Partial derivatives of the kernel with respect to train_x[i]:")
print(gradients)
