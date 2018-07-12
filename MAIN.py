# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 16:19:26 2018

@author: Quang Dien DUONG
"""

__author__ = 'Quang Dien Duong quangdien.duong[at]gmail.com'

from sparse_group_lasso import SGL
from cross_validation import CV
import numpy as np
import pickle
import time

#%%
with open('samples.pickle', 'rb') as handle:
    Y = pickle.load(handle)
    X = pickle.load(handle)
    knots = pickle.load(handle)
    splrep = pickle.load(handle)
    proj_matrix = pickle.load(handle)

u=0. # We test on the GDP samples
lbda1, alpha1, lbda2, alpha2 = np.array([0.04, 0.001, 0.04, 0.001])
K, p = knots.shape
c = np.ones((K-1)*p + 1)
coefs = np.concatenate((-c, c))
Y = np.asarray(Y)
#%%
sgl = SGL(Y, X, knots, u, lbda1, alpha1, lbda2, alpha2)
sgl.initialize_coefficient(coefs)
#gamma, sigma = sgl.fit_predict(sgl.X)
#%%
lbda = 0.04
alpha = 0.3
cv = CV(sgl, n_splits=3)
cv.initialize_parameters([lbda, alpha, lbda, alpha])
cv_out = cv.k_fold_cross_validation()
print(cv_out)
#%%
#lbda_rg = 10**np.r_[-3:1:11j]
#alpha_rg = np.r_[0.001:0.999:11j]
#resu = []
# Do some cross-validation
#for lbda in lbda_rg:
#    for alpha in alpha_rg:
#        start = time.time()
#        sgl_sample = sgl.copy()        
#        cv = CV(sgl_sample)
#        cv.initialize_parameters([lbda, alpha, lbda, alpha])
#        resu.append([lbda, alpha, cv.k_fold_cross_validation()])
#        end = time.time()
#        print("Running time for CV(lbda = %s, alpha = %s) is %s" % (lbda, alpha, end-start))
        
#with open('CV.pickle', 'wb') as handle:
#    pickle.dump(resu, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
#%%
#CV_out = None
#with open('CV.pickle', 'rb') as handle:
#    CV_out = pickle.load(handle)
    

