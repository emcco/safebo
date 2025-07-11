import torch
import gpytorch
import numpy as  np
from matplotlib import pyplot as plt
# from torchviz import make_dot
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
    B = globals.B
    alpha = torch.tensor([[ 1.0477],
        [-0.7144],
        [-1.9098],
        [ 1.1028],
        [ 1.3832]]).view(-1,1)
    
    # SE kernel
    # kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    # Matern kernel
    # kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=globals.nu))
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
    model.likelihood.noise = globals.noise_var
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
    noise_std_dev = torch.tensor(globals.noise_var)
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

def plot_true(true_x, true_y, jmin):
    
       
    jmin_np = np.empty(true_x.size(0))
    for i in range(len(jmin_np)):
        jmin_np[i] = jmin

    true_x_np = true_x.detach().numpy()
    true_y_np = true_y.detach().numpy()

    max_idx = torch.argmax(true_y)
    max_x = true_x_np[max_idx]
    max_y = true_y_np[max_idx]

    # Plot observed points
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1)

    # plt.figure(figsize=(10, 6))
    # ax.plot(true_x.detach().numpy(), true_y.detach().numpy(), 'g', label='True f, RKHS')
    ax.plot(true_x.detach().numpy(), true_y.detach().numpy(), 'b', label='True function f')

    # Plot line for jmin
    ax.plot(true_x.detach().numpy(), jmin_np, 'r-', label='Safety Threshold')
    ax.plot(max_x, max_y, 'bo', markersize=4, label='Global Max')
    ax.axvline(x=max_x, ymin=0, ymax=(max_y - plt.ylim()[0]) / (plt.ylim()[1] - plt.ylim()[0]), color='b', linestyle='dashed', linewidth=1.0)
    # ax.axhline(y=max_y, xmin=0, xmax=(max_x - plt.xlim()[0]) / (plt.xlim()[1] - plt.xlim()[0]), color= 'b', linestyle='dashed', linewidth=1.0)
    print(f'{(max_x - plt.xlim()[0]) / (plt.xlim()[1] - plt.xlim()[0])}')
    ax.axhline(y=max_y, xmin=0, xmax=0.788, color= 'b', linestyle='dashed', linewidth=1.0)

    ax.text(max_x + 0.5, plt.ylim()[0] - 0.05 * (plt.ylim()[1] - plt.ylim()[0]) + 0.3, rf"$x_{{\max}} = {max_x:.2f}$", color='b', fontsize=10, ha='center')
    ax.text(0.75, max_y+0.05, rf"$y_{{\max}} = {max_y:.2f}$", color='b', fontsize=10, ha='center')
    

    ax.set_xlabel('Input x', fontsize=12)
    ax.set_ylabel('Output y', fontsize=12)
    # ax.set_title('SafeOpt - Frequentist View', fontsize=18, fontweight='bold')
    ax.legend(loc='lower center')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)        # Save the plot
    ax.xaxis.set_major_locator(MultipleLocator(1.0))    # Major ticks every 1 unit
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))    # Minor ticks every 0.2 units
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(0,10)
    # Save the plot
    # string = 'comp' + globals.comp + '/comp' + globals.comp + '-' + str(globals.index) + '/plot' + "_" + 'turef' + ".png"
    string = 'true_RQ.png'
    fig.savefig(string, format='png')
    # plt.show()



def plot(model, true_x, true_y, test_x, jmin, b, x_sample, type, y_target, warning):
    
    mean, lower, upper = get_bounds_RKHS(model, test_x)
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
    ax.legend(loc='lower right', fontsize = 14)
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
    
    # iterations = np.arange(1, globals.iterations)
    iterations = np.arange(1, len(y_targets)+1)
    
    fig, ax = plt.subplots(figsize=(10,6))
    plt.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.1)

    # Plot the convergence line
    ax.plot(iterations, convergence, 'b', label='Convergence')
  
    
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(iterations)
    ax.set_xlim(1, len(y_targets)+1)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Regret R(t)')
    # ax.set_title('Convergence plot: SafeOpt - Frquentist View')
    ax.legend()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stringtime = str(timestamp)
    string = globals.folder_path + 'comp' +  globals.comp + '/' + globals.comp + '-' + str(globals.index) + '/plot_convergence' + globals.comp + '-' + str(globals.index) + '.png'
    fig.savefig(string, format='png')

def uncertainty_plot(max_w):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_uncertainty_data(max_w, timestamp)


    iterations = np.arange(1, globals.iterations)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(iterations, max_w, 'b', label='Max. uncertainty')
    ax.set_xticks(iterations)
    ax.axhline(0.05, color='red', linewidth=1)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Max. Uncertainty')
    ax.set_ylim(min(max_w) - 0.5, max(max_w) + 0.5)
    ax.set_title('Uncertainty plot: SafeOpt - Frequentist View')
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
    filename = f'{data_dir}/convergence_data_{globals.comp}-{globals.index}.json'
    with open(filename, 'w') as f:
        json.dump(data, f)



def get_observed_pred(model, test_x):
    model.eval()
    model.likelihood.eval()
    observed_pred = model(test_x) # likelihood kann weg
    return observed_pred    


def construct_uncertainty_width(model, x, R, delta, lambd):
    '''
        Acc. to Fiedler et al. 2021

        lamd: Likelihood noise
        delta: P >= 1 - delta
        R: observation noise
    '''
    # Read kernel and training data
    kernel = model.covar_module
    train_x = model.train_inputs[0]

    # Evaluate Gram matrix Kn and kn(x)
    Kn = kernel(train_x, train_x).evaluate()
    knx = kernel(train_x, x).evaluate()

    # Construct terms 
    Kn_reg = Kn + lambd * torch.eye(Kn.size(0), dtype=Kn.dtype)
    Kn_inv = torch.linalg.inv(Kn_reg)
    term1 = R * torch.norm(Kn_inv @ knx, dim=0)

    N = train_x.size(0)

    logterm = torch.log(1/delta)
    term2 = torch.sqrt(N + 2 * torch.sqrt(torch.tensor(N)) * torch.sqrt(logterm) + 2 * logterm)

    etaN = term1 * term2

    return etaN


def get_bounds_RKHS(model, x, B=globals.B, R=globals.R, delta=globals.delta, lambd=globals.lambd):

    observed_pred = get_observed_pred(model, x)
    eta_x = construct_uncertainty_width(model, x, R, delta, lambd)
    mean = observed_pred.mean
    stddev = observed_pred.stddev

    nomod = True

    if nomod:

        # lower = mean - stddev * (B + eta_x)
        # upper = mean + stddev * (B + eta_x)
        lower = mean - B * stddev - eta_x
        upper = mean + B * stddev + eta_x
      

        return mean, lower, upper
    else:
        alpha = torch.sqrt(torch.log(1 / delta))
        kernel = model.covar_module
        train_x = model.train_inputs[0]
      

        # Evaluate Gram matrix Kn
        Kn = kernel(train_x, train_x).evaluate()

        identity_matrix = torch.eye(Kn.size(0))
        K_reg = Kn + torch.square(torch.tensor([lambd])) * identity_matrix
        K_Sigma = Kn @ torch.linalg.inv(K_reg)
        Ks_fro = torch.linalg.norm(K_Sigma, ord='fro')
        Ks_2 = torch.linalg.norm(K_Sigma, ord=2)
        Ks_trace = torch.trace(K_Sigma)

        beta_sqrt = B + lambd * torch.sqrt(Ks_trace + 2 * alpha * Ks_fro + 2 * torch.square(alpha) * Ks_2)
        # beta_sqrt = B + torch.sqrt(Ks_trace + 2 * alpha * Ks_fro + 2 * torch.square(alpha) * Ks_2)

        lower = mean - beta_sqrt * stddev
        upper = mean + beta_sqrt * stddev

        return mean, lower, upper





class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # mean and kernel can be modified
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=globals.nu))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

        # self.covar_module._set_outputscale(0.6931471805599453) 
        # print(f'self.covar_module.base_kernel.lengthscale')

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    