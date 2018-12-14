# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:58:34 2018

@author: Quang Dien Duong
"""

__author__ = 'Quang Dien Duong quangdien.duong[at]gmail.com'

from scipy.stats import norm
import numpy as np


def generate_uniform_distribution_random_vector(n):
    resu = np.random.uniform(size = n)
    return resu

def generate_Burr_distributed_samples(eta, lamda, tau, size=1):
    assert eta > 0, "eta must be positive."
    assert lamda > 0, "lamda must be positive."
    assert tau > 0, "tau must be positive."
    assert type(size) == int, "size must be an integer."
    Fvec = generate_uniform_distribution_random_vector(size)
    resu = eta*((1-Fvec)**(-1/lamda)-1)
    resu = resu**(1/tau)
    return resu

# =============================================================================
#       Generalized Pareto distribution
# =============================================================================
def sigma_function(x):
    """
        x: array-type. In our setup, x is a tridimensional vector, i.e. x = [x1, x2, x3] 
    """
    resu = 0.1*(x[0]**4 + x[1]**4 + 10.0)
    return resu
    
def gamma_function(x):
    """
        x: array-type. In our setup, x is a tridimensional vector, i.e. x = [x1, x2, x3] 
    """
    resu = 0.15*(x[0]**2 - x[1]**2 + 4.0)
    return resu

def generate_GPD_samples(x, default_seed=1660):
    resu = []
    for i in range(len(x)):
        np.random.seed(default_seed+i)
        gamma = gamma_function(x[i])
        sigma = sigma_function(x[i])
        u = np.random.uniform()
        out = sigma/gamma*((1-u)**(-gamma)-1)
        resu.append(out)
    return resu