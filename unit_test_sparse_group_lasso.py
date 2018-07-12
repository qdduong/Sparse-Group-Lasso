# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:51:42 2018

@author: Quang Dien DUONG
"""

from sparse_group_lasso import SGL
from cross_validation import CV
import numpy as np
import utils as ut

from scipy.optimize import fmin_l_bfgs_b
__author__ = 'Quang Dien Duong quangdien.duong[at]gmail.com'


knots1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
knots2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
knots = np.array([knots1,knots2]).T

dx = 0.1
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

sgl = SGL(y0, X, knots, u, 0.1, 0.5, 0.05, 0.5)
sgl.initialize_coefficient(coefs)
sgl.define_groups()
#%%
#out=ut.partial_f_beta0(sgl.coef_, sgl.y, sgl.knots, sgl.splrep, sgl.proj_matrix, sgl.u
#sgl.fit()
#%%
cv = CV(sgl)
cv.param_ = np.array([sgl.lbda1, sgl.alpha1, sgl.lbda2, sgl.alpha2])
out = cv.k_fold_cross_validation()
