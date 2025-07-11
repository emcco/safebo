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
comp = 'SO_Freq_SE'

bool_bounds = True
# folder_path = 'code/Frequentist/SE/SafeOpt_Frequentist/'
folder_path = 'writing'