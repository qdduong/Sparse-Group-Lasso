# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:57:20 2018

@author: Quang Dien Duong
"""
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pickle
import time
import Local_Polynomial_Basis_Functions as LP


with open('samples.pickle', 'rb') as handle:
    X = pickle.load(handle)
    Y = pickle.load(handle)
    knots = pickle.load(handle)
    
#%%
#from sklearn.model_selection import train_test_split
#u = 1.1
#h = 3
#Xi = X[Y>=u]
#yi = Y[Y>=u] - u
#X_train, X_test, y_train, y_test = train_test_split(Xi, yi, test_size = 0.2, random_state=0)

#x = X_test[100]
#W = LP.get_weighting_coefs(x, X_train, h)    
#print(np.all(W == 0.))
#%%    
list_h = 10**np.arange(1, 2, 0.05)
list_u = np.arange(0.5, 1.6, 0.1)
# Initialize coefficient
d = len(X[0]) # d=10
degree = 1
poly = PolynomialFeatures(degree)
K = len(poly.fit_transform([list(np.ones(d))])[0])
one = np.ones(K)
onezero = np.zeros(K)
onezero[0] = 1.
init_coefs = np.array([-one, one])

out = []
for h in list_h:
    for u in list_u:
        start = time.time()
        Xi = X[Y>=u]
        yi = Y[Y>=u] - u   # Compute exceedances
        resu = LP.validation(Xi, yi, h, degree, init_coefs)
        out.append(resu)
        end = time.time()
        print("Running time for CV(h = %s, u = %s) is %s" % (h, u, end-start))
        
out = np.asarray(out)
out = out.reshape(len(list_h),len(list_u))
        
with open('CV_LP_d10_degree1_v3.pickle', 'wb') as handle:
    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%
#with open('CV_LP_d10_degree1.pickle', 'rb') as handle:
#    CV_LP_d10_degree1 = pickle.load(handle)
    
