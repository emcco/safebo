import gp_model as gpm
import optExp
import optMax
import numpy as np
import warnings
from botorch.exceptions import OptimizationWarning
import signal
import Timeouthandler
import torch



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

        # initial maximiser
        optimizer = optMax.OptMax(self.test_x, initial_model, self.jmin, self.Lk, self.Lmean)

        # if self.flag == False:
        #     with warnings.catch_warnings(record=True) as w:

        # try:
        #     signal.alarm(timeout_seconds)
        #     result_max = optimizer.bomin(self.x0_max)
        #     signal.alarm(0)
        # except Timeouthandler:
        #     print("Maximiser Optimization timed out!")

        #         # for warning in w:
        #         #     if issubclass(warning.category, OptimizationWarning):
        #         #         warn_string = "SLSQP timeout Max."
        #         #         print("Maximiser: OptimizationWarning encountered")
        #         #         break
        result_max = optimizer.bomin(self.x0_max)
        x_new_max = result_max[0].view(-1)
        mean, lower_max, upper_max = gpm.get_mean_bounds(initial_model, x_new_max, self.Lk, self.Lmean)
        var_max = upper_max - lower_max

        # Plot model+x_new_max for debugging
        # model_add_max, y_add_max = gpm.add_x(self.model, x_new_max)
        # Lk_max = gpm.comp_lip_kernel(model_add_max)
        # gpm.plot(model_add_max, Lk_max, torch.linspace(0, 10, 100), gpm.true_RKHS(torch.linspace(0, 10, 100))[0].squeeze(), self.test_x, jmin=-0.55, b=50, x_sample=x_new_max, type='model_add_max', y_target=y_add_max, warning=None)


        # initial expander
        optimizer_exp = optExp.OptExp(self.test_x, initial_model, self.jmin, self.Lk, self.Lmean)
        result_exp = None
        var_exp = None
        warn_string = None

        if self.flag == False:
            with warnings.catch_warnings(record=True) as w:
                
                result_exp = optimizer_exp.bomin(self.x0_exp)

                for warning in w:
                    if issubclass(warning.category, OptimizationWarning):
                        warn_string = "SLSQP timeout Exp."
                        print("Expander: OptimizationWarning encountered")
                        break 

            x_new_exp = result_exp[0]
            x_new_exp = x_new_exp[..., 0].view(-1)
            
            mean_exp, lower_exp, upper_exp = gpm.get_mean_bounds(initial_model, x_new_exp, self.Lk, self.Lmean)
            var_exp = upper_exp - lower_exp
        else:
            var_exp = 0
            print('Expander was set 0.')



        # determine new x_sample
        if var_max >= var_exp:
            x_sample = x_new_max
            x_sample_info = 'Maximiser'
        else:
            x_sample = x_new_exp
            x_sample_info = 'Expander'

        x_info = x_sample_info


        
        model_add, y_target = gpm.add_x(initial_model, x_sample)



        ### end: calc initial iteration ###

        return model_add, x_sample, x_info, y_target, warn_string
    
