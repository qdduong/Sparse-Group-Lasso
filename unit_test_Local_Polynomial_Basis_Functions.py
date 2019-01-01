# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:58:24 2018

@author: Quang Dien Duong
"""

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pickle

import Local_Polynomial_Basis_Functions as LP

yi = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
Xi = np.array([[0, 0.05],
               [0.2, 0.25],
               [0.4, 0.45],
               [0.6, 0.65],
               [0.8, 0.85],
               [1, 1.05]])
x = np.array([0.5, 0.5])

h = 0.6
u = 0.25

Xi = Xi[yi>=u]
yi = yi[yi>=u] - u
coefs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1., 1., 1., 1., 1., 1.])

resu = LP.get_neg_log_likelihood(coefs=coefs, x=x, Xi=Xi, yi=yi, bandwidth=h, degree=2) # Expected result is 0.926913

#%%
#def gamma0(x):
#    return 0.3 + 0.25*x + norm.pdf(x,-0.5,np.sqrt(0.01)) + norm.pdf(x,0.5,np.sqrt(0.05)) 

#with open('samples_test.pickle', 'rb') as handle:    
#    X = pickle.load(handle)
#    Y = pickle.load(handle)

#h = 0.3
#u = 1.1

#p = 1

#Xi = X[Y>=u]
#yi = Y[Y>=u] - u

#Xtest = np.linspace(-1., 1., num=100)
# Initialize coefficients
#poly = PolynomialFeatures(p)
#K = len(poly.fit_transform([list(np.ones(len(Xi[0])))])[0])
#one = np.array([1, 0.001])
#init_coefs = np.concatenate((-one, one))

#out=[]
#for x in Xtest:
#    args = (x, Xi, yi, h, p)
#    esti_gamma, esti_sigma, d = LP.optimization(init_coefs, args)
#    out.append(esti_gamma)
    
#out = np.asarray(out)

#gamma0 = gamma0(Xtest)
#plt.plot(Xtest, gamma0, label = "True")
#plt.plot(Xtest, out, label = "Estimated")
#plt.legend()
#plt.show()
# =============================================================================

#list_h = np.arange(0.1, 0.6, 0.05)
#list_u = np.arange(0.5, 2., 0.1)
#Xtest = np.linspace(-1., 1., num=100)

#gamma0 = gamma0(Xtest)
#plt.plot(Xtest, gamma0)
#plt.show()

#p = 1

#with open('samples_test.pickle', 'rb') as handle:    
#    X = pickle.load(handle)
#    Y = pickle.load(handle)

#AMSE = [] 
#for h in list_h:
#    for u in list_u:    
#        Xi = X[Y>=u]
#        yi = Y[Y>=u] - u
        # Initialize coefficients
#        poly = PolynomialFeatures(p)
#        K = len(poly.fit_transform([list(np.ones(len(Xi[0])))])[0])
#        one = np.array([1, 0.001])
#        init_coefs = np.concatenate((one, one))
#        out = []
#        for x in Xtest:
#            args = (x, Xi, yi, h, p)
#            esti_gamma, esti_sigma, d = LP.optimization(init_coefs, args, nb_max_iter=10000)
#            out.append(esti_gamma)
#        out = np.asarray(out)
#        resu = np.linalg.norm(out-gamma0,2)
#        AMSE.append(resu)

#AMSE = np.asarray(AMSE)
        
#with open('AMSE_1d_LP.pickle', 'wb') as handle:
#    pickle.dump(AMSE, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#resu = []
#listx = np.linspace(-0.8, 0.8, num=20, endpoint=True)
#for x in listx:
#    args = (x, Xi, yi, h, p)
#    esti_gamma, esti_sigma, d = optimization(init_coefs, args, nb_max_iter=10000)
#    resu.append(esti_gamma)
    
#plt.plot(listx, resu)
#plt.show()

