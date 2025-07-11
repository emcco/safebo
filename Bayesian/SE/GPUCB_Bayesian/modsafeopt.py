import gp_model as gpm
import optExp
import optMax
import numpy as np
import warnings
from botorch.exceptions import OptimizationWarning
import signal
import Timeouthandler
import torch
import safeopt as so



class ModSafeOpt:
    def __init__(self, x0_max, x0_exp, test_x, model, jmin, flag, Lk, Lmean):
        self.x0_max = x0_max
        self.x0_exp = x0_exp
        self.test_x = test_x
        self.model = model
        self.jmin = jmin
        self.iter_max = 50
        self.flag = flag
        self.Lk = Lk
        self.Lmean = Lmean




    def sample(self):

        # Timeout handler
        timeexep = Timeouthandler.TimeoutException()
        signal.signal(signal.SIGALRM, Timeouthandler.timeout_handler)
        timeout_seconds = 120


        initial_model = self.model
        x_sample = None
        x_sample_info = None


        ### start: calc initial iteration ###


        '''
        1. Create safe set
        2. Compute upper bound for all inputs in safeset
        3. Choose input with largest upper bound
        
        '''
    
        safe_set = so.classify_safe_set(initial_model, self.test_x, self.jmin, self.Lk, self.Lmean)
        n = safe_set.sum()
        ucb_set = torch.empty((0,2))
        for i in range(safe_set.size(0)):
            if safe_set[i] == True:

                upper = gpm.get_mean_bounds(initial_model, self.test_x[i].unsqueeze(-1), self.Lk, self.Lmean)[2]

                new_upper = torch.tensor([[i, upper]])
                ucb_set = torch.cat((ucb_set, new_upper), dim=0)

        uppers = ucb_set[:, 1]
        max_upper, max_index = torch.max(uppers, dim=0)
        upper_idx = ucb_set[max_index, 0]

        x_sample = self.test_x[int(upper_idx)].unsqueeze(-1)

        x_info = 'ucb'
        warn_string = None

        
        model_add, y_target = gpm.add_x(initial_model, x_sample)



        ### end: calc initial iteration ###

        return model_add, x_sample, x_info, y_target, warn_string
    
