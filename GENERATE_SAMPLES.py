# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:09:00 2018

@author: Quang Dien Duong
"""

__author__ = 'Quang Dien Duong quangdien.duong[at]gmail.com'


import setting as st
import numpy as np
import itertools as it
import random
import pickle

np.random.seed(1000)
N = 5
size = 5000

selected_inds = random.sample(range(N**10), size)
selected_inds.sort()

sple = np.linspace(0., 1., num=N, endpoint=False)

X = []
for x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 in it.product(sple,sple,sple,sple,sple,sple,sple,sple,sple,sple):
    X.append([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10])

X = np.asarray(X)
X = X[selected_inds]

# =============================================================================
# Generate knots
# =============================================================================
K = int(size**(1/5))
l1 = list(np.sort(np.random.uniform(size=K)))
l2 = list(np.sort(np.random.uniform(size=K)))
l3 = list(np.sort(np.random.uniform(size=K)))
l4 = list(np.sort(np.random.uniform(size=K)))
l5 = list(np.sort(np.random.uniform(size=K)))
l6 = list(np.sort(np.random.uniform(size=K)))
l7 = list(np.sort(np.random.uniform(size=K)))
l8 = list(np.sort(np.random.uniform(size=K)))
l9 = list(np.sort(np.random.uniform(size=K)))
l10 = list(np.sort(np.random.uniform(size=K)))
knots = np.array([l1,l2,l3,l4,l5,l6,l7,l8,l9,l10]).T
# =============================================================================
# Generate samples 
# =============================================================================
Y = []
lbda = 1
eta =1
for x in X:
    tau = 1/st.gamma_function(x)
    Y.append(float(st.generate_Burr_distributed_samples(eta, lbda, tau)))
    
Y = np.array(Y)

with open('samples.pickle', 'wb') as handle:
    pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(knots, handle, protocol=pickle.HIGHEST_PROTOCOL)
# =============================================================================
# True gamma and sigma function 
gamma0 = np.array([st.gamma_function(x) for x in X])
with open('gamma0.pickle', 'wb') as handle:
    pickle.dump(gamma0, handle, protocol=pickle.HIGHEST_PROTOCOL)
# =============================================================================

Xtest = np.array([[0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                  [0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  [0.12, 0.86, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
                  [0.76, 0.21, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
                  [0.22, 0.01, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15],
                  [0.92, 0.96, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15]])   
    

with open('out_of_samples.pickle','wb') as handle:
    pickle.dump(Xtest, handle, protocol=pickle.HIGHEST_PROTOCOL)
