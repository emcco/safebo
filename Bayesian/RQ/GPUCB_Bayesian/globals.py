import torch

noise = torch.tensor([0.001]) #likelihood noise

noise_stddev = torch.sqrt(noise)

# RKHS bound 
B = torch.tensor([4.0])
delta = torch.tensor([0.05])
Lf = torch.tensor([2.6591])
Lk = torch.tensor([0.4214])


# tools
iterations = 41 # actual iterations + 1

# PLotting

index = 1
comp = '2'
folder_path = 'code/Bayesian/RQ/GPUCB_Bayesian/'
# folder_path = 'extend79/'