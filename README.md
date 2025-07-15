# Safe Bayesian Optimization with Frequentist and Bayesian Error Bounds

Hi! Welcome to my git repo containing the software I wrote and used for my master thesis 'Safe Exploration Methods for Gaussian Processes' at the Institute of Control Systems at the University of Technology Hamburg, Germany. 

## Contents
- [Theory](#theory)
- [Challenges in Safe Bayesian Optimization](#challenges-in-safe-bayesian-optimization)
- [Research Question](#research-question)
- [Key Results](#key-results)
- [Software Implementation](#software-implementation)
- [Installation](#installation)
- [Notes](#notes)


## Theory

Bayesian Optimization Algorithms often rely on predictions about some unknown functions provided by a Gaussian Process (GP). GPs are a non-parametric, supervised machine learning method to predict the function value of each input by applying bayesian inference. The posterior prediction of a GP equips the user with a predicted function value ($\mu$) and a variance ($\sigma^2$) at each input. The hereby implemented SafeOpt and SafeUCB algorithms use those predictions to compute an uncertainty bound on each function value and suggests the subsequent input for practical evaluation to learn the true function tackling the trade-off between exploration and exploitation.

## Challenges in Safe Bayesian Optimization

Many practitioners use heuristics to construct such uncertainty bounds in the sense of $ \mu \pm B \cdot \sigma$. If the constant $B$ is heuristically chosen, there is no theoretical guarantee for the containment of the unknown true function. Thus, this work computes two types of theoretically sound approaches to achieve the containment of the true function with high probability. The frequentist error bounds assume the true function to be deterministic whereas the bayesian error bounds consider the true function to be a stochastic sample function from the GP. 

## Research Question

How conservative are the presented approaches of error bound construction and how applicable are they for real world settings? Are the differences kernel and/or algorithm dependent? 

## Key results

While the frequentist error bounds achieve superior algorithmic performance in terms of convergence rates, the theory behind requires an upper bound on the norm of the Reproducing Kernel Hilbert Space (RKHS) containing the true function which is not computable for an unknown function and therefore practically infeasible. This holds especially for safety-critical settings such as the medical field or human-robot-collaboration.  
The bayesian error bounds exhibit weaker performance, however their construction is solely data dependent and thus more applicable in real settings.

All results hold for the GP-UCB algorithm equivalently. 

## Software implementation 

- The presented code contains the implementation of the frequentist and bayesian view incorporating both, the Squared-Exponential (SE) and the rational-quadratic (RQ) kernel. Both approaches and kernel functions are each implemented in the SafeOpt and SafeUCB algorithms yielding in a total of eight single implementations. 
- Figures depicting the predicted function at various iteration steps as well as plotted convergence rates can be found too. 

## Author

**Emir Bajrović** ([@emcco](https://github.com/emcco))  
📍 Hamburg, Germany


![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.



## Installation 
For running one of the eight implementations depending on uncertainty bound construction (frequentist/bayesian), kernel choice (SE/RQ) and algorithm choice (SafeOpt/SafeUCB), pull the repo and run the according main file. The following shows an example of running the SafeOpt algorithm with the SE kernel and frequentist error bounds. 


```bash
git clone https://github.com/emcco/safebo.git
cd Frequentist/SE/SafeOpt_Frequentist
python main.py