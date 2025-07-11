import numpy as np
import math
import torch
import gp_model as gpm
# from scipy.optimize import minimize
# from autograd_minimize import minimize
from scipy.optimize import NonlinearConstraint
from botorch.generation.gen import gen_candidates_scipy


class OptMax:
    def __init__(self, test_x, model, jmin):
        
        self.jmin = jmin
        self.model = model
        self.test_x = test_x
        self.mean, self.lower, self.upper = gpm.get_bounds_RKHS(model, test_x)
        self.lower_max = torch.max(self.lower)


    def objective(self, x):
        
        if not x.requires_grad:
            x = x.detach().clone().requires_grad_(True)
        mean_max, lower_max, upper_max = gpm.get_bounds_RKHS(self.model, x)
        objective_val = upper_max - lower_max
        return objective_val


    def constraintMax(self, x):
        
        for i in range(x.dim()):
            if x.dim() > 2:
                x = x.squeeze(-1)
                if x.dim() < 3:
                    break

        upper_x = gpm.get_bounds_RKHS(self.model, x)[2]
        lower_max = self.lower_max
        c_maxi = upper_x - lower_max
        # gpm.plot(self.model, torch.linspace(0, 10, 100), gpm.true_RKHS(torch.linspace(0, 10, 100))[0].squeeze(), self.test_x, jmin=-0.55, b=2, x_sample=torch.tensor([0]), type='model_max', y_target=torch.tensor([0]), warning=None)
        return c_maxi
    
    def constraintSafety(self, x):

        # if not x.requires_grad:
        #     x = x.detach().clone().requires_grad_(True)
        # print(f'In OptMax/ConstraintSafety: x.requires_grad = {x.requires_grad}')

        # x = torch.tensor([x], dtype=torch.float32)
        # observed_pred = self.model(x)
        # lower_x = observed_pred.confidence_region()[0]
        
        for i in range(x.dim()):
            if x.dim() > 2:
                x = x.squeeze(-1)
                if x.dim() < 3:
                    break
                
        lower_x = gpm.get_bounds_RKHS(self.model, x)[1]
        if lower_x.size(0) > 1:
            print('Pause')
        jmin = self.jmin
        c_safe = lower_x - jmin
        return c_safe
    
    def bomin(self, x0):
    
        # torch.autograd.set_detect_anomaly(True)
        # print(f'In OptMax.py: x0.requires_grad = {x0.requires_grad}')

        nlc = [
            self.constraintSafety, 
            self.constraintMax
        ]

        # test_safe = self.constraintSafety(x0)
        # test_maxi = self.constraintMax(x0)

        x_min = torch.min(self.test_x)
        x_max = torch.max(self.test_x)

        bounds = torch.tensor([x_min, x_max])
        options = {
            "method": "SLSQP", 
            "maxiter": 1000
            }

        c_safe_initial = self.constraintSafety(x0)
        c_max_initial = self.constraintMax(x0)
        print('Check for MAX initial satifaction: c_safe %.3f, c_max: %.3f' % (
        c_safe_initial.item(), c_max_initial.item()))

        batch_candidates, batch_acq_values = gen_candidates_scipy(initial_conditions=x0, acquisition_function=self.objective, 
                                                                  lower_bounds=bounds[0], upper_bounds=bounds[1],
                                                                nonlinear_inequality_constraints=nlc, options=options)

        



        return batch_candidates, batch_acq_values
    
    # def min(self, x0):

        
    #     c_safe = self.constraintSafety
    #     c_maxi = self.constraintMax

   
    #     # nlc_safe = NonlinearConstraint(lambda x0 : c_safe(x0), 0, np.inf)
    #     # nlc_maxi = NonlinearConstraint(lambda x0 : c_maxi(x0), 0, np.inf)

    #     nlc_safe = {'type' : 'ineq', 'fun' : c_safe}
    #     nlc_maxi = {'type' : 'ineq', 'fun' : c_maxi}

    #     x_min = torch.min(self.test_x).item()
    #     x_max = torch.max(self.test_x).item()
        
    #     bounds = [(x_min, x_max)]

    #     # minimize from scipy 
    #     result = minimize(self.objective, x0, method='SLSQP', bounds=bounds)

    #     #minimize from autograd_scipy
    #     # result = minimize(self.objective, x0, backend='torch', method='SLSQP', constraints=[nlc_safe, nlc_maxi], bounds=bounds)

    

    #     return result

        
    
