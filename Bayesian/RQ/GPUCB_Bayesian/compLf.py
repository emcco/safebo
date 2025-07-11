import gp_model as gpm 
import torch 
import globals
import gpytorch

# true_x = torch.linspace(0, 10, 1000)
# true_y_sq = gpm.true_RKHS(true_x)[0]
# true_y = true_y_sq.squeeze()

true_x = torch.linspace(0, 10, 1000, requires_grad=True)

# Compute the RKHS function values
true_y = gpm.true_RKHS(true_x)[0]

# Compute the gradient (derivative) of the function w.r.t. x
gradients = torch.autograd.grad(true_y, true_x, grad_outputs=torch.ones_like(true_y), create_graph=True)[0]

# Compute the Lipschitz constant as the maximum absolute gradient
lipschitz_constant = torch.max(torch.abs(gradients)).item()

# Output the Lipschitz constant
print(f'Lf1 = {lipschitz_constant}')

y_max = torch.max(true_y)
y_min = torch.min(true_y)

x_max = true_x[torch.argmax(true_y)]
x_min = true_x[torch.argmin(true_y)]

Lf = (y_max - y_min)/(x_max - x_min)

print(f'Lf2 = {Lf}')


