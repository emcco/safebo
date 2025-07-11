import torch
import gpytorch
import numpy as  np
from matplotlib import pyplot as plt
import math
import globals
from datetime import datetime
from botorch.models import SingleTaskGP
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os
import json
import glob

def true_function(x, string):
    # return 10 * (x - 1) ** 2 * torch.exp(-0.4 * x)
    if string == 'lbfgs':
        return torch.exp(-0.4*x*10) * torch.sin(x*10)
        # output = torch.exp(-0.4*x) * torch.sin(x)
        # mean_output = torch.mean(output)
        # std_output = torch.std(output)
        # output_std = (output - mean_output) / std_output
        # return output_std
    else:
        return torch.exp(-0.4*x) * torch.sin(x)
    
def true_RKHS(x):

   # Requisits
    centers = torch.tensor([[5.2115],
        [5.4738],
        [1.8557],
        [7.8554],
        [3.4076]])
    # centers = globals.cente
    B = globals.B
    alpha = torch.tensor([[ 1.0477],
        [-0.7144],
        [-1.9098],
        [ 1.1028],
        [ 1.3832]]).view(-1,1)

    
    # alpha = globals.alpha
    # kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())


    # Evaluate Gram matrix of centers
    K = kernel(centers, centers).evaluate()
    # Compute and apply scaling
    scaling_factor = B / torch.sqrt(alpha.transpose(0,1) @ K @ alpha)
    scaled_alpha = scaling_factor * alpha

    # Evaluate function at x
    K_x_centers = kernel(x, centers).evaluate()
    rkhs_function = K_x_centers @ scaled_alpha

    norm = torch.sqrt(scaled_alpha.transpose(0,1) @ K @ scaled_alpha)

    return rkhs_function, norm



def init_model(train_x, train_y):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # likelihood.noise = torch.tensor([1e-2])
    model = ExactGPModel(train_x, train_y, likelihood)
    model.likelihood.noise = globals.noise
    return model


def create_singletask_gp(train_x, train_y):

    gp_model = SingleTaskGP(train_x, train_y)
    # gp_model.covar_module.base_kernel.lengthscale = torch.tensor([0.33035])
    # gp_model.likelihood.noise = torch.tensor([0.01533])
    # gp_model.covar_module.outputscale = torch.tensor([1.0])
    

    return gp_model


def add_x(model, x_add):
        
    if x_add.dim() > 1:
        for i in range(x_add.dim()):
            x_add = x_add[..., 0].squeeze(-1)
            if x_add.dim() < 2:
                break

    traingpm_x = model.train_inputs[0]
    traingpm_y = model.train_targets

    if traingpm_y.dim() == 0:
            traingpm_y = traingpm_y.unsqueeze(-1)

    train_x_add = torch.vstack([traingpm_x, x_add])
    # noise: noise_std_dev from main = math.sqrt(0.008)
    noise_std_dev = globals.noise_stddev
    y_add = true_RKHS(x_add)[0] + torch.randn(x_add.size()) * noise_std_dev
    y_add = y_add.squeeze(-1)
    train_y_add = torch.cat([traingpm_y, y_add])
    # train_y_add = true_function(train_x_add, 'nn') + torch.randn(train_x_add.size()) * math.sqrt(0.0006) # True function values
    train_y_add = train_y_add.squeeze()

    model_add = init_model(train_x_add, train_y_add)

    return model_add, y_add
    
def plot_params(model):
    print(f'Likelihood noise: {model.likelihood.noise.item()}')
    print(f'Mean constant: {model.mean_module.constant.item()}')
    print(f'Covariance outputscale: {model.covar_module.outputscale.item()}')
    print(f'Base kernel lengthscale: {model.covar_module.base_kernel.lengthscale.item()}')



def plot(model, Lk, Lmean, true_x, true_y, test_x, jmin, b, x_sample, type, y_target, warning):
    
    mean, lower, upper = get_mean_bounds(model, test_x, Lk, Lmean)
  
    train_x = model.train_inputs[0]
    train_y = model.train_targets

       
    jmin_np = np.empty(true_x.size(0))
    for i in range(len(jmin_np)):
        jmin_np[i] = jmin

    # Plot observed points
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.1)
    # plt.figure(figsize=(10, 6))
    # ax.plot(true_x.detach().numpy(), true_y.detach().numpy(), 'g', label='True f, RKHS')
    ax.plot(true_x.detach().numpy(), true_y.detach().numpy(), 'g', label='True f')

    # Plot line for jmin
    ax.plot(true_x.detach().numpy(), jmin_np, 'r-', label='Threshold')
    # Plot training data as black stars
    ax.plot(train_x.detach().numpy(), train_y.detach().numpy(), 'k*', label='Obsv.')
    # Plot predictive means as blue line
    ax.plot(test_x.detach().numpy(), mean.detach().numpy(), 'b', label='GP Mean')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, label='Uncertainty')
    # Added data point
    if not type == 'initial':
        ax.plot(x_sample.detach().numpy(), y_target.detach().numpy(), 'r*', label=f'{type}')

    test_x_np = test_x.detach().numpy()
    lower_np = lower.detach().numpy()
    
    above_threshold = lower_np > jmin
    x_highlight = test_x_np[above_threshold]

    # Add green bars where lower > jmin
    # if len(x_highlight) > 0:
    #     x_segments = np.split(x_highlight, np.where(np.diff(x_highlight) > 0.1)[0] + 1)
    #     for segment in x_segments:
    #         ax.axvspan(segment[0], segment[-1], color='green', alpha=0.3, label='Safe Region' if segment is x_segments[0] else "")

    ax.set_ylim(-5, 5)
    if len(x_highlight) > 0:
    # Split into contiguous segments if needed
        x_segments = np.split(x_highlight, np.where(np.diff(x_highlight) > 0.1)[0] + 1)
        
        for i, segment in enumerate(x_segments):
            # Draw a thick green line at y=0 (assuming that's your x-axis)
            ax.hlines(
                y=-5,                              # The y-value where the line will appear
                xmin=segment[0], 
                xmax=segment[-1],
                colors='green',
                linewidth=12,                     # Adjust thickness as desired
                alpha=0.6,                       # Transparency
                label='Safe Region' if i == 0 else ""  # Add label only once
            )
    # ax.plot([], [], '', label=warning)
    ax.set_xlabel('Input x', fontsize=12)
    ax.set_ylabel('Output y', fontsize=12)
    # ax.set_title('SafeOpt - Frequentist View', fontsize=18, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)        # Save the plot
    ax.xaxis.set_major_locator(MultipleLocator(1.0))    # Major ticks every 1 unit
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))    # Minor ticks every 0.2 units
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Save the plot
    string = globals.folder_path + 'comp' +  globals.comp + '/' + globals.comp + '-' + str(globals.index) + '/plot' + "_" + str(b+1) + ".png"
    directory = os.path.dirname(string)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(string, format='png')
    # plt.show()


def convergence_plot(y_max, y_targets):
    # Save the convergence data
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_convergence_data(y_max, y_targets, timestamp)
    
    # Plot (y_true_max - max_mean) with uncertainty bounds
    convergence = []
    for i in range(len(y_targets)):
        diff = y_max - y_targets[i]
        convergence.append(diff)
    convergence = np.array(convergence)
    iterations = np.arange(1, globals.iterations)
    
    fig, ax = plt.subplots(figsize=(10,6))
    # Plot the convergence line
    ax.plot(iterations, convergence, 'b', label='Convergence')
    
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(iterations)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('max(y_ture) - max(posterior mean)')
    ax.set_title('Convergence plot: SafeOpt - Bayesian View')
    ax.legend()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stringtime = str(timestamp)
    # string = 'comp' + globals.comp + '/comp' + globals.comp + '-' + str(globals.index) + '/plot_convergence' + '_' + stringtime + '.png'
    string = globals.folder_path + 'comp' +  globals.comp + '/' + globals.comp + '-' + str(globals.index) + '/plot_convergence' + globals.comp + '-' + str(globals.index) + '.png'

    fig.savefig(string, format='png')


def save_convergence_data(y_max, mean_values, timestamp):
    # Create directory if it doesn't exist
    data_dir = globals.folder_path + 'comp' + globals.comp + '/data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save the data
    data = {
        'y_max': y_max,
        'mean_values': mean_values,
        'timestamp': timestamp
    }
    # filename = f'{data_dir}/convergence_data_{timestamp}.json'
    filename = f'{data_dir}/convergence_data_{globals.comp}-{str(globals.index)}.json'

    with open(filename, 'w') as f:
        json.dump(data, f)

def uncertainty_plot(max_w):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_uncertainty_data(max_w, timestamp)


    iterations = np.arange(1, globals.iterations)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(iterations, max_w, 'b', label='Max. uncertainty')
    ax.set_xticks(iterations)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Max. Uncertainty')
    ax.set_ylim(min(max_w) - 0.5, max(max_w) + 0.5)
    ax.set_title('Uncertainty plot: SafeOpt - Bayesian View')
    ax.legend()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stringtime = str(timestamp)
    string = 'comp' + globals.comp + '/comp' + globals.comp + '-' + str(globals.index) + '/plot_uncertainty' + '_' + stringtime + '.png'
    fig.savefig(string, format='png')


def save_uncertainty_data(max_w, timestamp):
    data_dir = 'comp' + globals.comp + '/data_uncertainty'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save the data
    data = {
        'max_w': max_w,
        'timestamp': timestamp
    }
    filename = f'{data_dir}/uncertainty_data_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(data, f)





def get_observed_pred(model, test_x):
    model.eval()
    model.likelihood.eval()
    observed_pred = model(test_x) # likelihood kann weg
    return observed_pred    

def comp_lip_kernel(model):

    # train_x_model = model.train_inputs[0].clone().detach().requires_grad_(True)
    train_x_model = model.train_inputs[0].clone().requires_grad_(True)

    kernel = model.covar_module

    Lk = 0.0
    dot = None

    # kernel_matrix = kernel(train_x_model, train_x_model).evaluate_kernel().requires_grad_(True)
    kernel_matrix = kernel(train_x_model, train_x_model).evaluate_kernel()

    kernel_matrix = kernel_matrix.to_dense().requires_grad_(True)
    grad_outputs = torch.ones_like(kernel_matrix)



    grads = torch.autograd.grad(
         outputs=kernel_matrix,
         inputs=train_x_model,
         grad_outputs=grad_outputs,
         create_graph=True
    )[0]

    gradient_norm = torch.linalg.norm(grads)
    Lk = max(Lk, gradient_norm)


    # for i in range(N):
    #     for j in range(N):
    #         # Compute kernel value of xi, xj
    #         # k_eval = kernel(train_x_leaf[i], train_x_leaf[j]).evaluate().requires_grad_(True)
    #         # k_eval = kernel.forward(train_x_leaf[i], train_x_leaf[j]).requires_grad_(True)

    #         # Compute derivatives
    #         grad = torch.autograd.grad(
    #             k_eval,
    #             train_x_leaf[i],
    #             allow_unused=True
    #         )[0]

    #         # grad = k_trainx * ((train_x[i]-train_x[j])/length)

    #         if grad is not None:
    #             # Compute norm of the gradient
    #             gradient_norm = grad.norm()
    #             Lk = max(Lk, gradient_norm)  # Update maximum gradient norm
    #         else:
    #             print(f"Gradient is None for pair (train_x[{i}], train_x[{j}]): {train_x_leaf[i]}, {train_x_leaf[j]}")
    # # print(f'Lk = {Lk}')

    return Lk

def comp_post_mean_lip(model, Lk, train_x, train_y):

    # Lk = comp_lip_kernel(model)
    # train_x = model.train_inputs[0]

    if train_y.dim() == 0:
            train_y = train_y.unsqueeze(-1)

    N = torch.tensor([train_x.size(0)])
    # train_x = train_x_model.requires_grad_(True)
    # train_x_leaf = [torch.tensor(entry, requires_grad=True) for entry in train_x]

    kernel = model.covar_module
    K_dense = kernel(train_x, train_x).evaluate_kernel()
    K_reg = K_dense.to_dense()
    

    noise_stddev = globals.noise_stddev
    noise_var = noise_stddev ** 2
    K_noisy = K_reg + noise_var * torch.eye(N)
    K_inv = torch.linalg.inv(K_noisy)

    # Eigenvalues from K_inv and K_reg
    weight_norm = torch.sqrt(train_y.T @ K_inv @ K_reg @ K_inv @ train_y)
    # weights = K_inv @ train_y
    # weight_norm = torch.linalg.norm(weights)

    L_post_mean = Lk * torch.sqrt(N) * weight_norm
    if torch.isnan(L_post_mean).all():
        print(f'Lmean is NAN!')
    
    return L_post_mean

def compute_tau(model):
    train_x = model.train_inputs[0].view(-1)
    train_x_sorted, _ = torch.sort(train_x)
    spacings = torch.diff(train_x_sorted)
    tau = torch.max(spacings)
    return tau.unsqueeze(-1)

def compute_omega(model, tau, Lk, train_x):

    # Lk = comp_lip_kernel(model)
    # train_x = model.train_inputs[0]
    # N = torch.tensor([train_x.size(0)])

    # kernel = model.covar_module
    # K = kernel(train_x, train_x).evaluate()
    # noise_stddev = globals.noise_stddev
    # noise_var = noise_stddev ** 2

    # K_noisy = K + noise_var * torch.eye(N)
    # K_inv = torch.linalg.inv(K_noisy)
    # K_inv_norm = torch.norm(K_inv)

    # max_kernel_value = torch.max(K)
    

    # omega = torch.sqrt(2 * tau * Lk * (1 + N * K_inv_norm * max_kernel_value))
    omega = torch.sqrt(2 * tau * Lk)
    
    return omega
    
def compute_beta(model, tau, train_x_leaf):

    # train_x = model.train_inputs[0]
    # N = torch.tensor([train_x.size(0)])
    # input_range = torch.max(train_x_leaf) - torch.min(train_x_leaf)
    input_range = torch.tensor([10.0])
    Mtau = math.ceil(input_range / tau) + 1
    Mtau = torch.tensor([Mtau])
    delta = globals.delta
    beta = 2 * torch.log(Mtau / delta)
 
    return beta

# def compute_Lf_initial(test_x):

#     x = test_x.requires_grad_(True)
#     y = true_RKHS(test_x)[0]

#     grads = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]
    
#     Lf = torch.max(torch.abs(grads)).item()

#     return Lf 
    



def get_mean_bounds(model, test_x, Lk, Lmean):

    # Get training data
    train_x_model = model.train_inputs[0]

    train_xx = train_x_model.requires_grad_(True)
    test_x = test_x.requires_grad_(True)
    
   




    train_yy = model.train_targets
    if train_yy.dim() == 0:
            train_yy = train_yy.unsqueeze(-1)

    sort_ind = torch.argsort(train_xx.flatten())
    train_x = train_xx[sort_ind]
    train_y = train_yy[sort_ind]


    # Get mean, stddev
    observed_pred = get_observed_pred(model, test_x)
    mean = observed_pred.mean
    stddev = observed_pred.stddev

    # Hirche et al. parameters
    tau = torch.tensor([1/10000])
    beta = compute_beta(model, tau, train_x)
    Lf = globals.Lf
    Lk = globals.Lk
    omega = compute_omega(model, tau, Lk, train_x)
    gamma = (Lmean + Lf) * tau + torch.sqrt(beta) * omega
    
    # Compute bounds
    lower = mean - (torch.sqrt(beta) * stddev + gamma)
    upper = mean + (torch.sqrt(beta) * stddev + gamma)

    return mean, lower, upper





class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # mean and kernel can be modified
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

        # self.covar_module.base_kernel.lengthscale = torch.tensor([0.33578])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    