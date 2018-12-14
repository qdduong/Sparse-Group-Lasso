# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:28:02 2018

@author: Quang Dien Duong
"""

from scipy import misc
import numpy as np


def gradient_descent(func, x0, gradf, args=None, alpha=0.1, nb_max_iter = 100, eps = 1e-12):
    """
        x0: array-type
    """   
    z0 = func(x0, args)
    cond = eps + 10.0 # start with cond greater than eps (assumption)
    nb_iter = 0 
    tmp_z0 = z0
    x1_0 = x0
    while cond > eps and nb_iter < nb_max_iter:
        tmp_x1_0 = x1_0 - alpha * gradf(x1_0, args)        
        x1_0 = tmp_x1_0
        z0 = func(x1_0, args)
        nb_iter = nb_iter + 1
        cond = abs((tmp_z0 - z0)/z0)
        tmp_z0 = z0
    return x1_0, z0
