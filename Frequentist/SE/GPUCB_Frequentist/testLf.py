import globals
import math
epsilon = 0.2
K = 5
M = 10
d = 1
sigma = globals.noise_stddev**2

n = (sigma * M * d * K**(d+2)*math.log10(M*K/epsilon))/(epsilon**(d+4)) 
print(f'{n}')


