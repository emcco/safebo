import torch
import gpytorch
import gp_model as gpm
import safeopt as so
import globals
import modsafeopt as mso
import bayesianopt as bo
import os
import json 

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

    train_x = torch.tensor([6.5]).view(-1,1)
    # train_x = torch.linspace(0,10,5).view(-1,1)

    noise_std_dev = torch.sqrt(globals.noise_var)
    train_y = gpm.true_RKHS(train_x)[0] + torch.randn(train_x.size()) * noise_std_dev # True function values
    train_y = train_y.squeeze()

    # Initialize likelihood and model

    model = gpm.init_model(train_x, train_y)
    kernel = model.covar_module
    gpm.plot_params(model)

    # Make predictions on test data and plot
    test_x = torch.linspace(0, 10, 1000)       
                                         
    jmin = -0.5
 
    safe_set = so.classify_safe_set(model, test_x, jmin)
    safe_x = test_x[safe_set]

    mean, lower, upper = gpm.get_bounds_RKHS(model, test_x)

    x_sample_init = torch.tensor([0])
    y_target_init = torch.tensor([0])
    gpm.plot(model, true_x, true_y, test_x, jmin, b=-1, x_sample=x_sample_init, type='initial', y_target=y_target_init, warning=None)

    bo_optimizer = bo.BayesianOpt(test_x, jmin, train_x, train_y)
    y_targets_conv = []
    # max_w = []
    # json_filename = f"/Users/kaidahasanovic/Documents/mathesis_emir/code/SafeOpt_Frequentist/comp{globals.comp}/uncertainty_data.json"

    # Initialize an empty dictionary to store data
    # uncertainty_data = {"eta_x": [], "term1": [], "term2": []}


    for i in range(globals.iterations -1):
        print(f'Iteration: {i+1}')

        try:
            model, x_sample, add_type, y_target, warning = bo_optimizer.optimize(model)
        except Exception as e:
            filename = globals.folder_path + 'comp' +  globals.comp + '/' + globals.comp + '.txt'

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
        # Uncertainty data
        # mean, lower, upper = gpm.get_bounds_RKHS(model, test_x)
        # max_w.append(torch.max(upper-lower).item())
        # Plot current iteration
        if globals.index == 2:
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
    # gpm.uncertainty_plot(max_w)
    # gpm.plot_eta(json_filename)

    print('Done')


if __name__ == "__main__":
    
    globals.bool_bounds = True
    # globals.comp = 'SO_F_SE'
    globals.comp = 'plot'
    globals.index = 1
    for i in range(100):
        main()
        globals.index = globals.index + 1
