Code written during my Master Thesis "Safe and Efficient Exploration Methods for Gaussian Processes".

This repo contains the implementation of the Safe Bayesian Optimization Algorithm (SafeOpt) with modified, theoretically sound uncertainty bounds for the prediction of function output. 
Two different approaches are used, the bayesian and the frequentist view. The former assumes the true function to be a sample from the Gaussian Process, whereas the latter assumes the true function to be a deterministic funciton.

The aim of this work was to quantifiy the conservativeness of the GP prediction and thus algorithm performance when using the bayesian and frequentist uncertainty bounds. 

This thesis was written at the Insitute of Control Systems at the Unversity at Technology Hamburg, March 2025. 
