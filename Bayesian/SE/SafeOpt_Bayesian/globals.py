import torch

noise = torch.tensor([0.001]) #likelihood noise

noise_stddev = torch.sqrt(noise)

delta = torch.tensor([0.05])
Lf = 3.6821
Lk = 0.6065
B = torch.tensor([4])


# tools
iterations = 51 # actual iterations + 1

# PLotting

index = 1
comp = 'SO_B_SE'
# folder_path = 'code/Bayesian/SE/SafeOpt_Bayesian/'
folder_path = 'writing'