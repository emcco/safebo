import torch
import gpytorch
import gp_model as gpm
import safeopt as so
import globals
import modsafeopt as mso
import bayesianopt as bo
import os

def main():
    torch.set_default_dtype(torch.float64)


    # Generate true data from RKHS
    true_x = torch.linspace(0, 10, 100)
    true_y_sq, norm = gpm.true_RKHS(true_x)
    true_y = true_y_sq.squeeze()
    print("Norm: ", norm)
    true_y_max = max(true_y)
    y_max = true_y_max.item()

    # Generate training data

    train_x = torch.tensor([5.6]).view(-1,1)
    # train_x = torch.linspace(0,10,5).view(-1,1)

    noise_std_dev = torch.tensor(globals.noise_var)
    with torch.no_grad():
        train_y = gpm.true_RKHS(train_x)[0] + torch.randn(train_x.size()) * noise_std_dev # True function values
        train_y = train_y.squeeze()

    # Initialize likelihood and model

    model = gpm.init_model(train_x, train_y)
    kernel = model.covar_module.base_kernel

    # Print interpretable hyperparameters
    print(f"Lengthscale: {kernel.lengthscale.item()}")
    print(f"Outputscale (variance): {model.covar_module.outputscale.item()}")
    print(f'Alpha= {kernel.alpha.item()}')
    print(f'Likelihood noise: {model.likelihood.noise.item()}')
    # print(f"Nu: {kernel.nu}")  # This is the smoothness parameter, but it's not learned directly.
 

    # Make predictions on test data and plot
    test_x = torch.linspace(0, 10, 1000)       
                                         
    jmin = -0.5
 
    # gpm.plot_true(true_x, true_y, jmin)

    mean, lower, upper = gpm.get_bounds_RKHS(model, test_x)

    x_sample_init = torch.tensor([0])
    y_target_init = torch.tensor([0])
    safe_set = so.classify_safe_set(model, test_x, jmin)
    safe_x = test_x[safe_set]
    gpm.plot(model, true_x, true_y, test_x, jmin, b=-1, x_sample=x_sample_init, type='initial', y_target=y_target_init, warning=None)
    # gpm.plot_true(true_x, true_y, jmin)
   
    bo_optimizer = bo.BayesianOpt(test_x, jmin, train_x, train_y)
    y_targets_conv = []
    max_w = []

    for i in range(globals.iterations -1):
        print(f'Iteration: {i+1}')

        try:
            model, x_sample, add_type, y_target, warning = bo_optimizer.optimize(model)
        except Exception as e:
            
            filename = globals.folder_path + 'comp' +  globals.comp + '/' + globals.comp + '.txt'

            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)

            message = 'Failure in execution - comp' + str(globals.comp) + '-' + str(globals.index) + '- at iteration: ' + str(i+1) + '\n'
            with open(filename, "a") as file:
                file.write(message)
            print(f'Error encountered in iteration {i+1}: {e}')
            print(f'Iterations cut at {i+1}')
            break

        # Convergence data
        y_targets = model.train_targets
        y_targets_max = max(y_targets)
        y_targets_conv.append(y_targets_max.item())
  
        if globals.index == 1:
            gpm.plot(model, true_x, true_y, test_x, jmin, i, x_sample, add_type, y_target, warning)
        if globals.index == 40:
            gpm.plot(model, true_x, true_y, test_x, jmin, i, x_sample, add_type, y_target, warning)
        if globals.index == 60:
            gpm.plot(model, true_x, true_y, test_x, jmin, i, x_sample, add_type, y_target, warning)
        if globals.index == 80:
            gpm.plot(model, true_x, true_y, test_x, jmin, i, x_sample, add_type, y_target, warning)
        if globals.index == 100:
            gpm.plot(model, true_x, true_y, test_x, jmin, i, x_sample, add_type, y_target, warning)
        print('----------------------------------------------------')

    gpm.convergence_plot(y_max, y_targets_conv)


    print('Done')


if __name__ == "__main__":
    


    # globals.comp = 'SO_F_RQ3'
    globals.comp = 'plot'
    globals.bool_bounds = True
    globals.index = 1
    for i in range(100):
        main()
        globals.index = globals.index + 1
