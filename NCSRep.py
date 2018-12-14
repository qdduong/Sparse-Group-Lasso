# -*- coding: utf-8 -*-

import numpy as np

def remove_duplicate(knots):
    knots = list(knots)
    return np.asarray(list(set(knots)))

def expansion(array, size):
    """
        array: 1D array-type
        size : integer
    """
    assert type(size) == int, "size must be an integer"
    array = np.asarray(array)
    out = np.ones((len(array),size))
    for i in range(len(out)):
        out[i] = array[i]*out[i]
    return out

def get_d_function(x, knots):
    """
        x: 1d-array-type
    """
    knots = remove_duplicate(knots)
    out = []
    for k in knots[:-1]:
        out.append([(max(i-k,0)**3 - max(i-knots[-1],0)**3)/(knots[-1]-k) for i in x])
    return out

def get_NCS_basis_function(x, knots):
    """
        x, knots: 1D-array-type variables 
    """
    out = []
    knots.sort()
    h1 = list(np.ones(len(x)))
    h2 = list(x)
    out.append(h1)
    out.append(h2)
    dfuncs = get_d_function(x,knots)
    for k in range(len(dfuncs)-1):
        temps = list(np.asarray(dfuncs[k]) - np.asarray(dfuncs[-1]))
        out.append(temps)
    return out
        




    
    
