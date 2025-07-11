import torch


noise_var = torch.tensor([0.001])

# RKHS bound 
B = torch.tensor([4.0])
delta = torch.tensor([0.05])
lambd = noise_var
R = noise_var



iterations = 21 # actual iterations + 1

# PLotting

index = 1
comp = 'test1'
# folder_path = 'code/Frequentist/SE/GPUCB_Frequentist/'
folder_path = 'writing/'