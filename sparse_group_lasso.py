# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:53:04 2018

@author: Quang Dien Duong
"""

from utils import l2, get_splrep, get_regfunc, get_projecting_matrix, return_group_to_index, discard_group, block_wise_descent_fitting, get_coef_exclude_ind, gradient_f_beta0, gradient_f_theta0
from scipy.optimize import fmin_l_bfgs_b
import numpy as np


__author__ = 'Quang Dien Duong quangdien.duong[at]gmail.com'


class SGL(object):
    def __init__(self, y, X, knots, u, lbda1, alpha1, lbda2, alpha2, max_iter=1000, rtol=1e-6):
        self.y = y
        self.X = X
        self.knots = knots
        self.splrep = get_splrep(self.X, self.knots)
        self.proj_matrix = get_projecting_matrix(self.knots)
        self.u = u
        self.lbda1 = lbda1
        self.alpha1 = alpha1
        self.lbda2 = lbda2
        self.alpha2 = alpha2
        self.max_iter = max_iter
        self.rtol = rtol
        self.coef_ = None
        
    def initialize_coefficient(self, init_values):
        self.coef_ = np.asarray(init_values)
        
    def define_groups(self):
        self.groups = return_group_to_index(self.coef_, self.knots)
        
    def copy(self):
        return self
        
    def fit(self):
        # sparse group lasso selection on kappa
        distance = 1.
        nb_iter = 1
        while distance > self.rtol and nb_iter <= self.max_iter:
            coefs_old = self.coef_.copy()
            for gr in self.groups[0][1:]:
            # 1- Should the group be zero-ed out?
                tmp_coefs_gr = self.coef_.copy()
                tmp_coefs_gr[gr] = 0.
                if discard_group(self.y, self.knots, tmp_coefs_gr, self.splrep, self.proj_matrix, self.u, self.lbda1, self.alpha1, gr):
                    self.coef_[gr] = 0.
                # 2- If the group is not zero-ed out, update each component
                else:
                    self.coef_[gr] = block_wise_descent_fitting(self.coef_, self.y, self.knots, self.splrep, self.proj_matrix, self.u, self.lbda1, self.alpha1, gr)
                         
            
            # sparse group lasso selection on tau
            for gr in self.groups[1][1:]:
            # 1- Should the group be zero-ed out?
                tmp_coefs_gr = self.coef_.copy()
                tmp_coefs_gr[gr] = 0.
                if discard_group(self.y, self.knots, tmp_coefs_gr, self.splrep, self.proj_matrix, self.u, self.lbda2, self.alpha2, gr):
                    self.coef_[gr] = 0.
                # 2- If the group is not zero-ed out, update each component
                else:
                    self.coef_[gr] = block_wise_descent_fitting(self.coef_, self.y, self.knots, self.splrep, self.proj_matrix, self.u, self.lbda2, self.alpha2, gr)
                    
            # estimate beta_0
            ind_beta0 = int(self.groups[0][0])
            beta0_old = self.coef_[ind_beta0]
            coef_excl_beta0 = get_coef_exclude_ind(self.coef_, ind_beta0)
            beta0_new, ignored1, ignored2 = fmin_l_bfgs_b(func=gradient_f_beta0, x0=beta0_old, 
                                      args=(coef_excl_beta0, ind_beta0, self.y, self.knots, self.splrep, self.proj_matrix, self.u), approx_grad=True) 
            self.coef_[ind_beta0] = beta0_new
            # estimate theta_0
            ind_theta0 = int(self.groups[1][0])
            theta0_old = self.coef_[ind_theta0]
            coef_excl_theta0 = get_coef_exclude_ind(self.coef_, ind_theta0)
            theta0_new, ignored3, ignored4 = fmin_l_bfgs_b(func=gradient_f_theta0, x0= theta0_old,
                                       args=(coef_excl_theta0, ind_theta0, self.y, self.knots, self.splrep, self.proj_matrix, self.u), approx_grad=True)
            self.coef_[ind_theta0] = theta0_new
            # update_nb_iter
            nb_iter += 1
            # update_distance
            distance = l2(self.coef_ - coefs_old)
        return self
        
    def predict(self, x):
        splrep_x = get_splrep(x, self.knots, self.X)
        coefs_reshap = self.coef_.reshape((2, int(len(self.coef_)/2)))
        beta = coefs_reshap[0]
        theta = coefs_reshap[0]
        gamma = np.exp(get_regfunc(beta, splrep_x, self.proj_matrix))
        sigma = np.exp(get_regfunc(theta, splrep_x, self.proj_matrix))
        return gamma, sigma
       
    def fit_predict(self, x):
        self.define_groups()
        return self.fit().predict(x)