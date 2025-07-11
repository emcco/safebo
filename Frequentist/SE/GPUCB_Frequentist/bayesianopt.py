import modsafeopt
import gp_model as gpm
import safeopt as so
import torch
import numpy as np
import random
'''
1. Update safe_set
2. Update max_set
3. Find IC
4. Initialize SafeOpt optimizer


'''

class BayesianOpt:
    def __init__(self, test_x, jmin, train_x, train_y):
        
        self.test_x = test_x
        self.jmin = jmin
        self.train_x = train_x
        self.train_y = train_y

    def optimize(self, model):

        mean, lower, upper = gpm.get_bounds_RKHS(model, self.test_x)
        safe_set= so.classify_safe_set(model, self.test_x, self.jmin)
        lower_max = torch.max(lower)
        max_set = so.classify_maximizer(model, self.test_x)

        x0_max = so.get_maximiser_ic(self.test_x, safe_set, max_set)
        x0_exp = so.get_expander_ic(self.test_x, safe_set, model)
        # x0_exp = so.get_expander_ic_mod()



        # Check if slope around lower bound @ x_new_exp is small enough
        x_exp_unsafe = x0_exp[..., 1].view(-1)
        x_exp_safe = x0_exp[..., 0].view(-1)

      
      

        flag = False
   

        optimizer = modsafeopt.ModSafeOpt(x0_max, x0_exp, self.test_x, model, self.jmin, flag)
        result = optimizer.sample()
        model_add = result[0]
        x_sample = result[1]
        x_info = result[2]
        y_target = result[3]
        warning = result[4]



        print(f'Next x: {x_sample.item()}')
        print(f'Type: {x_info}')

        

        return model_add, x_sample, x_info, y_target, warning
        