import numpy as np
import globals
import torch
import gp_model as gpm
from scipy.optimize import minimize
from botorch.generation.gen import gen_candidates_scipy


class OptExp:
    def __init__(self, test_x, model, jmin, Lk, Lmean):
        self.test_x = test_x
        self.jmin = jmin
        self.model = model
        self.Lk = Lk
        self.Lmean = Lmean
        self.mean, self.lower, self.upper = gpm.get_mean_bounds(model, test_x, self.Lk, self.Lmean)
        self.lower_max = torch.max(self.lower)

    
    def constraintSafety(self, x):
        # x_safe = torch.tensor([x[0,0,0]])
        x_safe = x[..., 0].unsqueeze(-1)
        lower_x = gpm.get_mean_bounds(self.model, x_safe, self.Lk, self.Lmean)[1]
        jmin = self.jmin
        c_safe = lower_x - jmin
        return c_safe
    
    def constraintUnsafe(self, x):
        x_unsafe = x[..., 1].unsqueeze(-1)
        lower_x = gpm.get_mean_bounds(self.model, x_unsafe, self.Lk, self.Lmean)[1]
        jmin = self.jmin
        c_unsafe = jmin - lower_x
        return c_unsafe
    
    def constraintExpander(self, x):
        x_safe = x[..., 0].unsqueeze(-1)
        x_unsafe = x[..., 1].unsqueeze(-1)

        if x_safe.dim() > 1:
            for i in range(x_safe.dim()):
                x_safe = x[..., 0].squeeze(-1)
                x_unsafe = x[..., 1].squeeze(-1)

                if x_safe.dim() < 2:
                    break



        # get obs. pred @ unsafe x
        # get upper(x_safe)
        upper = gpm.get_mean_bounds(self.model, x_safe, self.Lk, self.Lmean)[2]
        y_add = upper
        x_add = x_safe

        traingpm_x = self.model.train_inputs[0]
        traingpm_y = self.model.train_targets

        if traingpm_y.dim() == 0:
            traingpm_y = traingpm_y.unsqueeze(-1)

        train_x_add = torch.vstack([traingpm_x, x_add])
        train_y_add = torch.cat([traingpm_y, y_add])

        
        model_add = gpm.init_model(train_x_add, train_y_add)
        model_add.likelihood.noise = globals.noise        
        model_add.eval()
        # gpm.plot(model_add, torch.linspace(0, 10, 100), gpm.true_RKHS(torch.linspace(0, 10, 100))[0].squeeze(), self.test_x, jmin=-0.55, b=1, x_sample=x_safe, type='model_add_exp', y_target=y_add, warning=None)

        lower_add = gpm.get_mean_bounds(model_add, x_unsafe, self.Lk, self.Lmean)[1]

        jmin = self.jmin
        c_exp = lower_add - jmin
        return c_exp
    
    def objective(self, x):

        # x_safe = torch.tensor([x[0,0,0]], requires_grad=True)
        x_safe = x[..., 0].unsqueeze(-1)
        mean, lower, upper = gpm.get_mean_bounds(self.model, x_safe, self.Lk, self.Lmean)        
        objective_val = upper - lower 
        return objective_val

    
 
    
    def bomin(self, x0):
    

        nlc = [
            self.constraintSafety, 
            self.constraintUnsafe, 
            self.constraintExpander
        ]


        x_min = torch.min(self.test_x)
        x_max = torch.max(self.test_x)

        bounds = torch.tensor([x_min, x_max])
        options = {
            "method": "SLSQP", 
            "maxiter": 100
            }
        
        c_safe_initial = self.constraintSafety(x0)
        c_unsafe_initial = self.constraintUnsafe(x0)
        c_exp_initial = self.constraintExpander(x0)

        print('Check for EXP initial satifaction: c_safe %.3f, c_unsafe: %.3f, c_exp: %.3f' % (
        c_safe_initial.item(), c_unsafe_initial.item(), c_exp_initial.item()))

        batch_candidates, batch_acq_values = gen_candidates_scipy(initial_conditions=x0, acquisition_function=self.objective, 
                                                                  lower_bounds=bounds[0], upper_bounds=bounds[1],
                                                                nonlinear_inequality_constraints=nlc, options=options)

        return batch_candidates, batch_acq_values
    
       

        
    
