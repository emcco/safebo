import globals
import torch
import gp_model as gpm




K = torch.tensor([[6.9315e-01, 1.0377e-03, 3.4813e-12, 5.8399e-13, 2.6175e-26, 0.0000e+00],
        [1.0377e-03, 6.9315e-01, 1.0377e-03, 4.1875e-04, 3.4813e-12, 2.6174e-26],
        [3.4813e-12, 1.0377e-03, 6.9315e-01, 6.7292e-01, 1.0377e-03, 3.4813e-12],
        [5.8399e-13, 4.1875e-04, 6.7292e-01, 6.9315e-01, 2.4234e-03, 1.9559e-11],
        [2.6175e-26, 3.4813e-12, 1.0377e-03, 2.4234e-03, 6.9315e-01, 1.0377e-03],
        [0.0000e+00, 2.6174e-26, 3.4813e-12, 1.9559e-11, 1.0377e-03, 6.9315e-01]])



K_lin = torch.tensor([[6.9315e-01, 1.0377e-03, 0, 0, 0, 0.0000e+00],
        [1.0377e-03, 6.9315e-01, 1.0377e-03, 4.1875e-04, 0, 0],
        [0, 1.0377e-03, 6.9315e-01, 6.7292e-01, 1.0377e-03, 0],
        [0, 4.1875e-04, 6.7292e-01, 6.9315e-01, 2.4234e-03, 0],
        [0, 0, 1.0377e-03, 2.4234e-03, 6.9315e-01, 1.0377e-03],
        [0.0000e+00, 0, 0, 0, 1.0377e-03, 6.9315e-01]])









train_x = torch.linspace(0, 10, 5).view(-1,1) 
noise_std_dev = globals.noise_stddev
train_y = gpm.true_RKHS(train_x)[0] + torch.randn(train_x.size()) * noise_std_dev # True function values
train_y = train_y.squeeze()
model = gpm.init_model(train_x, train_y)
true_x = torch.linspace(0, 10, 100)
true_y_sq, norm = gpm.true_RKHS(true_x)
true_y = true_y_sq.squeeze()
jmin = -0.55
x_sample_init = torch.tensor([0])
y_target_init = torch.tensor([0])

test_x = torch.linspace(0, 10, 1000)

train_x_model = model.train_inputs[0].requires_grad_(True)
kernel = model.covar_module

kernel_matrix = kernel(train_x_model).evaluate_kernel()
kernel_matrix = kernel_matrix.to_dense()
grad_outputs = torch.ones_like(kernel_matrix)


grads = torch.autograd.grad(
    outputs=kernel_matrix,
    inputs=train_x_model,
    grad_outputs=grad_outputs
)
print(f'grads: {grads}')

Lk1 = gpm.comp_lip_kernel(model, None)

gpm.plot(model, true_x, true_y, test_x, jmin, b=-1, x_sample=x_sample_init, type='initial', y_target=y_target_init, warning=None)


x_add = torch.tensor([4.358])

model_add = gpm.add_x(model, x_add)[0]
train_x_model = model_add.train_inputs[0].requires_grad_(True)
kernel = model_add.covar_module

kernel_matrix = kernel(train_x_model, train_x_model).evaluate_kernel()
kernel_matrix = kernel_matrix.to_dense()
grad_outputs = torch.ones_like(kernel_matrix)


grads = torch.autograd.grad(
    outputs=kernel_matrix,
    inputs=train_x_model,
    grad_outputs=grad_outputs
)

gpm.plot(model_add, true_x, true_y, test_x, jmin, b=-1, x_sample=x_sample_init, type='initial', y_target=y_target_init, warning=None)
Lk2 = gpm.comp_lip_kernel(model_add, None)


print(f'grads: {grads}')