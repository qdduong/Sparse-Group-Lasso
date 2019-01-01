# -*- coding: utf-8 -*-
"""
Author Quang Dien DUONG
"""

import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import fmin_l_bfgs_b, approx_fprime

from sklearn.model_selection import train_test_split


def get_epanechnikov_kernel(u):
    u = np.asarray(u)
    resu = 3./4*(1-u**2)
    resu[np.abs(u) > 1] = 0
    return resu

#%%
def get_local_polynomial_basis_functions(x, Xi, degree):
    """
        x: (1,d) array type
        Xi: (n,d) array type where n is the sample size
        degree: maximum degree of local polynomial 
    """
    x = np.asarray(x)
    Xi = np.asarray(Xi)
    Delta_x = Xi - x
    poly = PolynomialFeatures(degree)
    out = poly.fit_transform(Delta_x)
    return out

def get_weighting_coefs(x, Xi, bandwidth):
    """
        x: (1,d) array type
        Xi: (n,d) array type where n is the sample size
        bandwidth: positive number
    """
    x = np.asarray(x)
    Xi = np.asarray(Xi)
    Delta_x_over_bandwidth = norm(Xi - x, ord=2, axis=1)/bandwidth
    resu = get_epanechnikov_kernel(Delta_x_over_bandwidth)/bandwidth
    return resu


#%%    
def get_neg_log_GDP_density(yi, gamma, sigma):
    #if not np.all(sigma > 0.) or not np.all(1. + gamma*yi/sigma > 0.):
    #    print('stop')
    return np.log(sigma) + (1. + 1./gamma) * np.log(1. + gamma*yi/sigma)
 

def get_neg_log_likelihood(coefs, x, Xi, yi, bandwidth, degree):
    tmp_coefs = coefs
    tmp_coefs = tmp_coefs.reshape(2,int(len(tmp_coefs)/2))
    basis_funcs = get_local_polynomial_basis_functions(x, Xi, degree)
    weighting = get_weighting_coefs(x, Xi, bandwidth)
    
    gamma = np.exp(np.sum(tmp_coefs[0]*basis_funcs, axis=1))
    sigma = np.exp(np.sum(tmp_coefs[1]*basis_funcs, axis=1))
    neg_log_GDP = get_neg_log_GDP_density(yi, gamma, sigma)
    return np.mean(neg_log_GDP*weighting)


#%%
# Optimization process
def optimization(coef0, args):
    """
        coef0: array-type
        args = (x, Xi, yi, bandwidth, degree)
    """
    minimizer, fmin, d = fmin_l_bfgs_b(func=get_neg_log_likelihood,
                                       x0=coef0,
                                       args=args,
                                       approx_grad=True)
    minimizer = minimizer.reshape(2,int(len(minimizer)/2))
    esti_gamma = np.exp(minimizer[0][0])
    esti_sigma = np.exp(minimizer[0][0])
    return esti_gamma, esti_sigma, d    


def optimization0(coef0, args, alpha=0.1, nb_max_iter = 1000, eps = 1e-6):
    """
        coef0: array-type
        args = (x, Xi, yi, bandwidth, degree)
    """
    x, Xi, yi, bandwidth, degree = args
    z0 = get_neg_log_likelihood(coef0, x, Xi, yi, bandwidth, degree)
    cond = eps + 10.0 # start with cond greater than eps (assumption)
    nb_iter = 0 
    tmp_z0 = z0
    x1_0 = coef0
    while cond > eps and nb_iter < nb_max_iter:
        gradf = approx_fprime(x1_0, get_neg_log_likelihood, 1e-6, x, Xi, yi, bandwidth, degree)
        tmp_x1_0 = x1_0 - alpha * gradf        
        x1_0 = tmp_x1_0
        z0 = get_neg_log_likelihood(x1_0, x, Xi, yi, bandwidth, degree)
        nb_iter = nb_iter + 1
        cond = abs((tmp_z0 - z0)/tmp_z0)
        tmp_z0 = z0
    tmp_coefs = x1_0.reshape(2,int(len(x1_0)/2))
    esti_gamma = np.exp(tmp_coefs[0][0])
    esti_sigma = np.exp(tmp_coefs[1][0])
    d = (x1_0, z0, cond, nb_iter)
    return esti_gamma, esti_sigma, d

def validation(Xi, yi, bandwidth, degree, coef0 = None, test_size=0.2, n_splits=5):
    X_train, X_test, y_train, y_test = train_test_split(Xi, yi, test_size = test_size, random_state=0)
    out = []
    for xt, yt in zip(X_test, y_test): 
        args = (xt, X_train, y_train, bandwidth, degree)
        gamma_xt, sigma_xt, d = optimization(coef0, args)
        out.append(get_neg_log_GDP_density(yt, gamma_xt, sigma_xt))
    out = np.asarray(out)
    return np.mean(out)    
    