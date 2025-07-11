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

        # safe_set, unsafe_set = so.classify_safe_set(lower, self.jmin)
        safe_set = so.classify_safe_set(model, self.test_x, self.jmin)
        max_set = so.classify_maximizer(model, self.test_x)

        x0_max = so.get_maximiser_ic(self.test_x, safe_set, max_set)
        x0_exp = so.get_expander_ic(model, self.test_x, safe_set)
        # x0_exp = so.get_expander_ic_mod()



        # Check if slope around lower bound @ x_new_exp is small enough
        x_exp_unsafe = x0_exp[..., 1].view(-1)
        x_exp_safe = x0_exp[..., 0].view(-1)

      
        lower_safe = gpm.get_bounds_RKHS(model, x_exp_safe)[1]         

        lower_unsafe = gpm.get_bounds_RKHS(model, x_exp_unsafe)[1]
      

        flag = False
        # if torch.abs(m).item() > 0.45: # if slope too large, choose maximiser 
        #     flag = True




        # Find the positions of each value in x0_exp within test_x
        positions = torch.empty(2)
        for k in range(2):
            for j in range(len(self.test_x)):
                if self.test_x[j].item() == x0_exp[..., k].item():
                    positions[k] = j

        if safe_set[int(positions[0].item())] != True:
            print('Error: Initial condition of expander opt. not fulfilled! Supposed x_safe is not safe!')

        if safe_set[int(positions[1].item())] != False:
            print('Error: Initial condition of expander opt. not fulfilled! Supposed x_unsafe is safe!')

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
        