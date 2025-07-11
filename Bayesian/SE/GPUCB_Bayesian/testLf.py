import torch 
import globals
import gpytorch
import gp_model as gpm
import numpy as np
import math



def kernel_rq(r, alpha, sigma, l):
    return sigma*(1 + (r**2) / (2 * alpha * l**2))**(-alpha)

def kernel_se(r, sigma, l):
    return sigma*(torch.exp(-(r**2)/(2*(l**2))))

'''
Compute Lf,SE
'''
f1 = torch.tensor([1.45])
f2 = torch.tensor([-1.75])
x1 = torch.tensor([3.6])
x2 = torch.tensor([1.8])
Ltest = torch.abs(f1-f2)/torch.abs(x1-x2)
r = torch.linspace(0,10,1000, requires_grad=True)
y_true = gpm.true_RKHS(r)[0]
y_grads = torch.autograd.grad(y_true, r, grad_outputs=torch.ones_like(y_true), create_graph=True)[0]
LfSE = torch.max(torch.abs(y_grads)).item()


alpha = torch.tensor([0.6931])
l = alpha
sigma = torch.tensor([0.6931])
# sigma = alpha


r = torch.linspace(0,10,1000, requires_grad=True)

output = kernel_rq(r, alpha, sigma, l)
# output = kernel_se(r, sigma, l)



r_grads = torch.autograd.grad(output, r, grad_outputs=torch.ones_like(output), create_graph=True)[0]
Lk = torch.max(torch.abs(r_grads)).item()


print('Done')


true_x = torch.linspace(0, 10, 1000, requires_grad=True)
true_y = gpm.true_RKHS(true_x)[0]
# true_y = true_y_sq.squeeze()

Lf_grads = torch.autograd.grad(true_y, true_x, grad_outputs=torch.ones_like(true_y), create_graph=True)[0]
Lf = torch.max(torch.abs(Lf_grads)).item()

print('Done')