import torch

noise = torch.tensor([0.001]) #likelihood noise

noise_stddev = torch.sqrt(noise)

# RKHS bound 
B = torch.tensor([4.0])
delta = torch.tensor([0.05])
Lf = 3.6821
Lk = 0.6065


# tools
iterations = 21 # actual iterations + 1

# PLotting
index = 1
comp = 'test'
folder_path = 'code/Bayesian/SE/GPUCB_Bayesian/'
