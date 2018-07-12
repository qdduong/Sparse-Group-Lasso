# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 18:12:13 2018

@author: Quang Dien Duong
"""

import numpy as np
import utils as ut
import scipy as sc
from scipy import optimize

__author__ = 'Quang Dien Duong quangdien.duong[at]gmail.com'

knots1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
knots2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
knots = np.array([knots1,knots2]).T

dx = 0.01
x = np.arange(0,7,dx)
X = np.concatenate((x,x)).reshape(2,len(x)).T

# =============================================================================
# Construct coef
# =============================================================================
K, p = knots.shape
n = len(x)
coef0 = np.ones((K-1)*p + 1)

l1, l2, l3, l4 = [0, 0, 0, 0]

P = ut.get_projecting_matrix(knots)
splrep = ut.get_splrep(X, knots)

# =============================================================================
# Test objective_func and grad_func function 
# =============================================================================

y0 = np.arange(1,8,dx)
u=3.0
coefs = np.concatenate((-0.02*coef0, coef0))
objfunc = ut.loss(coefs, y0, knots, splrep, P, u, 0, 0, 0, 0)
#%%
print("objection function =",objfunc)
gradfunc = np.round(ut.grad_func(coefs, y0, knots, splrep, P, u),2)
print()
print("---- Test grad_func function by comparing it with approx_fprime function -----")
print("grad-func = ", gradfunc)
def func(coefs):
    return ut.loss(coefs, y0, knots, splrep, P, u, 0, 0, 0, 0)  
approx_fprime = np.round(optimize.approx_fprime(coefs, func, epsilon=1e-6),2)
print("approx-fprime = ", approx_fprime)
print("Is grad-func equal to approx-fprime? :", np.array_equal(gradfunc, approx_fprime))
#%%

print()
print("-----Test return_group_to_index--------")
groups = ut.return_group_to_index(coefs, knots)
print(groups)
#%%
z0 = np.arange(9)-4
l = 1
print("-----Test coordinate-wise soft thresholding operator---------")
print("z0 = ", z0)
print("lambda = ", l)
print("S(z0,lamda) =", ut.coordinate_wise_soft_thresholding(z0, l))

#%%
indices_group = groups[0][2]
alpha = 0.5
lbda = 0.1
z = ut.grad_func(coefs, y0, knots, splrep, P, u)[indices_group]                      
print("-----Test discard_group-------")
print("print coordinate-wise soft thresholding operator: ", ut.coordinate_wise_soft_thresholding(z, alpha*lbda))
print(ut.discard_group(y0, knots, coefs, splrep, P, u, lbda=lbda, alpha=alpha, indices_group=indices_group))
#%%
print()
print("----Test F function ----")
print("coefs = ", coefs)
print("F(coefs, 1) = ", ut.F(coefs, 1, y0, knots, splrep, P, u, lbda, alpha, indices_group))
#%%
print()
print("----Test optimize_step_size-----")
print("t0 = 1")
t_optimal = ut.optimize_step_size(coefs, y0, knots, splrep, P, u, lbda, alpha, indices_group)
print("t_optimal = ", t_optimal)
#%%
print()
print("----Test block_wise_descent_fitting----")
coefs_group_optimal = ut.block_wise_descent_fitting(coefs, y0, knots, splrep, P, u, lbda, alpha, indices_group)
print("resu = ",coefs_group_optimal)
#%%
#print()
#print("-----Test get_coef_exclude_ind------------")
#tmp = np.arange(1,11)
#print("Before excluding the fourth element")
#print(tmp)
#print("After excluding the fourth element")
#tmp1 = ut.get_coef_exclude_ind(tmp, 3)
#print(tmp1)
#print("Insert '4' into the excluded position")
#print(ut.get_coef_include_ind(tmp1, 4, 3))
#%%
#print()
#print("------Test obj_func-----------------")
#print("Objective function calculated by ut.loss = %s", objfunc)
#coef_excl_beta0 = ut.get_coef_exclude_ind(coefs, 0)
#beta0 = coefs[0]
#objfunc2 = ut.obj_func(beta0, coef_excl_beta0, 0, y0, knots, splrep, P, u, 0, 0, 0, 0)
#print("Objective function calculate by ut.obj_func = %s", objfunc2)
#%%
#print()
#print("------Test gradient_f_beta0--------")
#print("gradfunc[0] = ", gradfunc[0])
#gradien_f_beta0 = ut.gradient_f_beta0(beta0, coef_excl_beta0, 0, y0, knots, splrep, P, u)
#print("gradien_f_beta0 = ", gradien_f_beta0)
#%%
