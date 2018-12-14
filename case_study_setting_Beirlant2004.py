# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:06:09 2018

@author: Quang Dien DUONG
"""

__author__ = 'Quang Dien Duong quangdien.duong[at]gmail.com'

import numpy as np
import setting as st
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm


np.random.seed(1000)

N = 2000

X = np.linspace(-1., 1., num=N, endpoint = True)

def tau(x):
    return 1./(0.3 + 0.25*x + norm.pdf(x,-0.5,np.sqrt(0.01)) + norm.pdf(x,0.5,np.sqrt(0.05))) 

Y = []
for x in X:
    Y.append(float(st.generate_Burr_distributed_samples(eta=1, lamda=1, tau=tau(x))))

X = np.asarray(X).reshape(len(X),1)    
Y = np.asarray(Y)

with open('samples_test.pickle', 'wb') as handle:
    pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
# unit_test
#gamma = [1./tau(x) for x in X]
#plt.plot(X, gamma)









