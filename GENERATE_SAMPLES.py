# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:09:00 2018

@author: Quang Dien Duong
"""

__author__ = 'Quang Dien Duong quangdien.duong[at]gmail.com'


import setting as st
import numpy as np
import pickle

np.random.seed(1000)
N = 10

X1 = np.linspace(0.0, 1.0, num=N, endpoint=False)
X2 = np.linspace(0.0, 1.0, num=N, endpoint=False)
X3 = np.linspace(0.0, 1.0, num=N, endpoint=False)

X = []
for x1 in X1:
    for x2 in X2:
        for x3 in X3:
            X.append([x1,x2,x3])
X = np.round(np.asarray(X),2)
# =============================================================================
# Generate knots
# =============================================================================
l1 = list(np.sort(np.random.uniform(size=34)))
l2 = list(np.sort(np.random.uniform(size=34)))
l3 = list(np.sort(np.random.uniform(size=34)))
knots = np.array([l1,l2,l3]).T
# =============================================================================
# Generate samples 
# =============================================================================
Y = st.generate_GPD_samples(X)
# =============================================================================
# True gamma and sigma function 
gamma0 = np.array([st.gamma_function(x) for x in X])
sigma0 = np.array([st.sigma_function(x) for x in X])
with open('gamma0_sigma0.pickle', 'wb') as handle:
    pickle.dump(gamma0, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(sigma0, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('samples.pickle', 'wb') as handle:
    pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(knots, handle, protocol=pickle.HIGHEST_PROTOCOL)
# =============================================================================    