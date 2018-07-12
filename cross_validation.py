# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:23:35 2018

@author: Quang Dien DUONG
"""

__author__ = 'Quang Dien Duong quangdien.duong[at]gmail.com'

from sparse_group_lasso import SGL
from sklearn.model_selection import KFold
from utils import get_splrep, l
import numpy as np

__author__ = 'Quang Dien Duong quangdien.duong[at]gmail.com'

class CV(object):
    def __init__(self, sgl,  n_splits=5):
        assert type(sgl) == SGL, "sgl must be a Sparse_group_lasso object"
        self.sgl = sgl
        self.n_splits = n_splits
        self.kf = KFold(n_splits=self.n_splits)
        self.kf.get_n_splits(self.sgl.X)
        self.param_ = None
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.splrep_train = []
        for train_index, test_index in self.kf.split(self.sgl.X):
            X_train, X_test = self.sgl.X[train_index], self.sgl.X[test_index]
            y_train, y_test = self.sgl.y[train_index], self.sgl.y[test_index]
            self.X_train.append(X_train)
            self.X_test.append(X_test)
            self.y_train.append(y_train)
            self.y_test.append(y_test)   
            self.splrep_train.append(get_splrep(X_train, self.sgl.knots))
            
    def initialize_parameters(self, param):
        self.param_ = param
            
    def k_fold_cross_validation(self):
        lbda1, alpha1, lbda2, alpha2 = self.param_
        resu=[]
        for i in range(self.n_splits):
            sgl_i = SGL(self.y_train[i], self.X_train[i], self.sgl.knots, self.sgl.u, lbda1, alpha1, lbda2, alpha2)
            K, p = self.sgl.knots.shape
            cf = np.ones((K-1)*p + 1)
            init_coefs = np.concatenate((-cf, cf))
            sgl_i.initialize_coefficient(init_coefs)
            gamma_i, sigma_i = sgl_i.fit_predict(self.X_test[i])
            resu.append(np.mean(l(self.y_test[i], gamma_i, sigma_i)))
        resu = np.asarray(resu)
        return np.mean(resu)
            
    
