# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 16:19:26 2018

@author: Quang Dien DUONG
"""

__author__ = 'Quang Dien Duong quangdien.duong[at]gmail.com'

from sparse_group_lasso import SGL
from cross_validation import CV
from mpl_toolkits.mplot3d import Axes3D
from utils import f, surface
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import optimize
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
    
with open('gamma0_sigma0.pickle', 'rb') as handle:
    gamma0 = pickle.load(handle)
    sigma0 = pickle.load(handle)
    
#%%
# Plot gamma0
N = 10
X1 = np.linspace(0.0, 1.0, num=N, endpoint=False)
X2 = np.linspace(0.0, 1.0, num=N, endpoint=False)
X1, X2 = np.meshgrid(X1, X2)
G0 = 0.5*(X1**2 - X2**2 + 1.4)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, G0, cmap=cm.coolwarm, linewidth=0, antialiased=True)
# Customize the z axis
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Gamma0')
#%%
# Plot Sigma0
S0 = 10*(X1**4 + X2**4 + 2.0)

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(X1, X2, S0, cmap=cm.coolwarm, linewidth=0, antialiased=True)
# Customize the z axis
ax2.zaxis.set_major_locator(LinearLocator(10))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig2.colorbar(surf2, shrink=0.5, aspect=5)
plt.title('Sigma0')
plt.show()

#%%
u=0. # We test on the GDP samples
lbda1, alpha1, lbda2, alpha2 = np.array([1e-03, 0.01, 1e-04, 0.01])
K, p = knots.shape
c = np.ones((K-1)*p + 1)
coefs = np.concatenate((-c, 3*c))
Y = np.asarray(Y)
#%%
sgl = SGL(Y, X, knots, u, lbda1, alpha1, lbda2, alpha2)
sgl.initialize_coefficient(coefs)
gamma, sigma = sgl.fit_predict(sgl.X)
#%%
# Mean squared error
#lbda_rg = 10**np.r_[-3.5:0:11j]
#mu_rg = 10**np.r_[-3.5:0:11j]
#alpha = 0.1
#resu = []
#for lbda in lbda_rg:
#    for mu in mu_rg:
#        start = time.time()
#        sgl = SGL(Y, X, knots, u, lbda, alpha, mu, alpha)
#        sgl.initialize_coefficient(coefs)
#        gamma, sigma = sgl.fit_predict(sgl.X)
#        resu.append(np.mean((gamma - gamma0)**2) + np.mean((sigma - sigma0)**2))
#        end = time.time()
#        print("Running time for MSE(lbda = %s, mu = %s) is %s" % (lbda, mu, end-start))
        
#resu = np.array(resu).reshape((len(lbda_rg), len(mu_rg)))
        
#with open('MSE0_alpha0_1.pickle', 'wb') as handle:
#    pickle.dump(resu, handle, protocol = pickle.HIGHEST_PROTOCOL)
#%% 
# -----------------------------
# Export MSE_******.pickle file
# -----------------------------   
#MSE = None
#with open('MSE0_alpha0_1.pickle', 'rb') as handle:
#    MSE = pickle.load(handle)
    
#lbda_X, mu_Y = np.meshgrid(np.r_[-3.5:0:11j], np.r_[-3.5:0:11j])
#plt.figure()
#CS = plt.contour(lbda_X, mu_Y, MSE*100)
#if plt.rcParams["text.usetex"]:
#    fmt = r'%r'
#else:
#    fmt = '%r'
#plt.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
    

#%%
#lbda = 0.04
#alpha = 0.3
#cv = CV(sgl, n_splits=3)
#cv.initialize_parameters([lbda, alpha, lbda, alpha])
#cv_out = cv.k_fold_cross_validation()
#print(cv_out)
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
    

    
