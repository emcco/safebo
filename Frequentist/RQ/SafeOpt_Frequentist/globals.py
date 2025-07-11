import torch

noise_var = torch.tensor([0.001])

# RKHS bound 
B = torch.tensor([4.0])
delta = torch.tensor([0.05])
lambd = noise_var
R = noise_var


# tools
iterations = 51 # actual iterations + 1


# PLotting

index = 1
comp = 'writing'

bool_bounds = True
nu = 2.5
# folder_path = 'code/Frequentist/RQ/SafeOpt_Frequentist/'
folder_path = 'writing'