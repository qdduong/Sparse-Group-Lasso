# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:12:21 2018

@author: Quang Dien Duong
"""

import numpy as np
import NCSRep as ncs

__author__ = 'Quang Dien Duong quangdien.duong[at]gmail.com'

#%%

def l1(x):
    """
        x: array
        
    Return norm L1 of an array x
    """
    return np.linalg.norm(x,1)

def l2(x):
    """
        x: array
        
    Return norm L2 of an array x
    """
    return np.linalg.norm(x,2)

def surface(coef, X, Y):
    return coef[0] + coef[1]*X + coef[2]*Y + coef[3]*(X**2) + coef[4]*(X*Y) + coef[5]*(Y**2) + coef[6]*(X**3) + coef[7]*(X**2)*Y + coef[8]*X*(Y**2) + coef[9]*(Y**3)


def f(coef, X, Y, Z):
    Z0 = surface(coef, X, Y)
    out = (Z0-Z)**2
    return np.mean(out)

#%%

def get_splrep(x, knots, Xi=None):
    """
    This function is meant to build the centered natural cubic spline basic function given the corresponding "knots" and the predictors "Xi".
        Xi: predictors
        x : variables 
    
    Return set of the centered spline representation
    """
    x = np.asarray(x)
    if Xi is None:
        Xi = x
    else:    
        Xi = np.asarray(Xi)
    knots = np.asarray(knots)
    n, p = x.shape
    out = []
    for j in range(p):
       splrepj = np.asarray(ncs.get_NCS_basis_function(x.T[j], knots.T[j]))
       # =====================================================
       # Take into account centering effect
       centered_vect = np.asarray(ncs.get_NCS_basis_function(Xi.T[j], knots.T[j])).mean(axis=1)
       centered_basis = ncs.expansion(centered_vect, n)
       splrepj = splrepj - centered_basis
       out.append(splrepj)
    return out

def get_projecting_matrix(knots):
    """
    This function is meant to creat a collection of the projecting matrix.
    For example, the predictors live on a p-dimensional vector space. We denote by P the projectin matrix.
    Then the product P[i]*coeficient will return the coeficient vector in the i-th direction.
    
    Return set of the projecting matrix
    """
    knots = np.asarray(knots)
    K, p = knots.shape
    vect0 = np.zeros(K-1).reshape(K-1,1)
    mat0 = np.zeros((K-1,K-1))
    I = np.eye(K-1)
    Pmatrix = []
    for i in range(p):
        Pi = vect0
        for j in range(p):
            if i==j:
                Pi = np.concatenate((Pi,I), axis=1)
            else:
                Pi = np.concatenate((Pi,mat0), axis=1)
        Pmatrix.append(Pi)
    return np.asarray(Pmatrix)

def get_regfunc(coefs, splrep, proj_matrix):
    d = len(splrep)
    P = proj_matrix
    out = coefs[0]
    for j in range(d):
        coefj = np.dot(P[j],coefs)
        splrepj = splrep[j][1:]
        out = out + np.dot(coefj, splrepj)
    return out

def get_P_nL(y, betas, thetas, splrep, proj_matrix, u=0.0):
    """
        By default, u is set to 0, which corresponds to the case where y exactly follows the GDP
    """
    assert u >= 0.0 
    y = np.asarray(y)
    y = y - u
    kappa = get_regfunc(betas, splrep, proj_matrix)
    tau = get_regfunc(thetas, splrep, proj_matrix)
    resu = tau + (1+np.exp(-kappa))*np.log(1+np.exp(kappa-tau)*np.maximum(y, 0.0))
    resu[np.where(y < 0.0)] = 0.0
    return np.mean(resu)

def get_penalty(betas, thetas, proj_matrix, lbda1, alpha1, lbda2, alpha2):
    """
        Get the penalty term with lbda_1 <-- (1-alpha1)*lbda1
                                  lbda_2 <-- alpha1*lbda1
                                  mu_1   <-- (1-alpha2)*lbda2
                                  mu_2   <-- alpha2*lbda2
    """
    d = len(proj_matrix)
    resu1 = 0.
    resu2 = 0.
    for gr in range(d):
        G = len(proj_matrix[gr])
        resu1 += np.sqrt(G)*l2(np.dot(proj_matrix[gr],betas))
        resu2 += np.sqrt(G)*l2(np.dot(proj_matrix[gr],thetas))
    resu3 = l1(betas[1:])
    resu4 = l1(thetas[1:])
    return (1-alpha1)*lbda1*resu1 + alpha1*lbda1*resu3 + -(1-alpha2)*lbda2*resu2 + alpha2*lbda2*resu4

def l(y, gamma, sigma):
    """
        Get the Kullback-Leibler divergence in terms of the response variables 'y', 'gamma' and 'sigma' which are functions of the predictors 'Xi'.
        This function is used in the cross validation selection. 
    """
    return np.log(sigma)+(1.+1./gamma)*np.log(1. + gamma*y/sigma)

def loss(coefs, y, knots, splrep, proj_matrix, u, lbda1, alpha1, lbda2, alpha2):
    """
        Get the objective loss function in terms of the fitting coefficients.
        In this method, 'y','knots','splrep','proj_matrix', 'u', 'lbda1', 'alpha1', 'lbda2', 'alpha2' are paramters.
    """
    coefs = np.asarray(coefs)
    knots = np.asarray(knots)
    coefs = coefs.reshape((2,int(len(coefs)/2)))
    betas = coefs[0]
    thetas = coefs[1]
    K, p = knots.shape
    y = np.asarray(y)
    # =========================================================================
    # Consistency conditions
    assert len(betas) == 1+(K-1)*p, "Dimensional mismatch on beta."
    assert len(thetas) == 1+(K-1)*p, "Dimensional mismatch on theta."
    assert lbda1 >= 0.0, "lbda1 must be positive."
    assert lbda2 >= 0.0, "lbda2 must be positive."
    assert 1.0 >= alpha1 >= 0.0, "alpha1 must be included in the interval [0,1]."
    assert 1.0 >= alpha2 >= 0.0, "alpha2 must be included in the interval [0,1]."
    # =========================================================================
    P_nL = get_P_nL(y, betas, thetas, splrep, proj_matrix, u)
    pen = get_penalty(betas, thetas, proj_matrix, lbda1, alpha1, lbda2, alpha2)
    return P_nL + pen

def grad_func(coefs, y, knots, splrep, proj_matrix,u):
    """
        Get gradient vector of the loss function 
    """
    coefs = np.asarray(coefs)
    knots = np.asarray(knots)
    coefs = coefs.reshape((2,int(len(coefs)/2)))
    betas = coefs[0] # First half of coefs is bates
    thetas = coefs[1] # Second half of coefs is thetas
    K, p = knots.shape
    y = np.asarray(y)
    y = y-u

    kappa = get_regfunc(betas, splrep, proj_matrix)
    tau = get_regfunc(thetas, splrep, proj_matrix)
    resu = []
    # =========================================================================
    # partial_F/ partial_beta_1
    dF_dbeta1 = np.exp(-kappa)*np.log(1+np.exp(kappa-tau)*np.maximum(y,0.0))-(1+np.exp(-kappa))*(np.exp(kappa-tau)*np.maximum(y,0))/(1+np.exp(kappa-tau)*np.maximum(y,0.0))
    dF_dbeta1[np.where(y < 0.0)] = 0.0
    resu.append(-np.mean(dF_dbeta1))
    
    # =========================================================================
    # partial_F/ partial_beta_j
    for j in range(p):
        splrepj = splrep[j][1:] # Takeaway the first constant row
        # =====================================================
        betaj = np.dot(proj_matrix[j],betas)
        for k in range(len(betaj)):
            dF_dbetajk = splrepj[k] * (np.exp(-kappa) * np.log(1 + np.exp(kappa-tau) * np.maximum(y,0.0)) - (1 + np.exp(-kappa)) * (np.exp(kappa-tau) * np.maximum(y,0.0))/(1 + np.exp(kappa-tau) * np.maximum(y,0.0)))
            dF_dbetajk[np.where(y < 0.0)] = 0.0
            resu.append(-np.mean(dF_dbetajk))
            
    # =========================================================================
    dF_dtheta1 = -1 + (1+np.exp(-kappa))*(np.exp(kappa-tau)*np.maximum(y,0.0))/(1+np.exp(kappa-tau)*np.maximum(y,0.0))
    dF_dtheta1[np.where(y < 0.0)] = 0.0
    resu.append(-np.mean(dF_dtheta1))
    
    # =========================================================================
    # partial_F/ partial_theta_j
    for j in range(p):
        splrepj = splrep[j][1:] # Takeaway the first constant row
        # =====================================================
        thetaj = np.dot(proj_matrix[j],thetas)
        for k in range(len(thetaj)):
            dF_dthetajk = splrepj[k]*(-1 + (1+np.exp(-kappa))*(np.exp(kappa-tau)*np.maximum(y,0.0))/(1+np.exp(kappa-tau)*np.maximum(y,0.0)))
            dF_dthetajk[np.where(y < 0.0)] = 0.0
            resu.append(-np.mean(dF_dthetajk))
    return np.asarray(resu)

def return_group_to_index(coefs, knots):
    """
        The coefficient vector will be divided into groups.
        This method is meant to seperate the coefficient vector into the set of groups of index.  
    """
    coefs = np.asarray(coefs)
    knots = np.asarray(knots)
    K,p = knots.shape
    out = [np.array([0])]
    for i in range(p):
        tmp = np.arange(i*(K-1) + 1, (i+1)*(K-1)+1)
        out.append(tmp)
    out.append(np.array([p*(K-1)+1]))
    for i in range(p):
        tmp = np.arange((i+p)*(K-1) + 2, (i+p+1)*(K-1) + 2)
        out.append(tmp)
    out = np.asarray(out).reshape((2,int(len(out)/2)))
    return out

#def discard_group(y, knots, coefs, splrep, proj_matrix, u, lbda, alpha, indices_group):
#    gradf_group = grad_func(coefs, y, knots, splrep, proj_matrix, u)[indices_group]
#    Gj = len(indices_group)
#    tk = []
#    for k in range(Gj):
#        tmp = abs(gradf_group[k]/(alpha * lbda))
#        if tmp <= 1.0:
#            tk.append(- gradf_group[k]/(alpha * lbda))
#        else:
#            tk.append(- np.sign(gradf_group[k]/(alpha * lbda)))       
#    tk = np.asarray(tk)
#    J = 1/(Gj*(1-alpha)*lbda)**2 * np.linalg.norm(gradf_group + alpha*lbda*tk,2)
#    return J<=1.0

#def discard_component(y, knots, coefs, splrep, proj_matrix, u, lbda, alpha, ind_sparse):
#    gradf_ind = grad_func(coefs, y, knots, splrep, proj_matrix, u)[ind_sparse]
#    return abs(gradf_ind) <= alpha*lbda
#%%
def coordinate_wise_soft_thresholding(z, lbda):
    """
    The coordinate-wise soft thresholding operator is defined as S(z,lbda)_i = sign(z)_i * max(abs(z_i) - lbda, 0)
    where z is a vector and lbda is a postive constant.
        z: array
        lbda: positive number
    
    Return S(z,lbda)
    """
    z = np.asarray(z)
    sign_z = np.sign(z)
    z1 = np.abs(z) - lbda
    z1[np.where(z1<0.0)] = 0.0
    return sign_z*z1


def discard_group(y, knots, coefs, splrep, proj_matrix, u, lbda, alpha, indices_group):
    """
    Check if || S(gradf(coefs_group)[indices_group] , alpha * lbda) ||_2 <= (1-alpha) * lbda * sqrt(G_j)
    
    Return: Boolean 
    """
    gradf_group = grad_func(coefs, y, knots, splrep, proj_matrix, u)[indices_group]
    G_j = len(indices_group)
    S = coordinate_wise_soft_thresholding(gradf_group, alpha*lbda)
    return l2(S) <= (1-alpha)*lbda*np.sqrt(G_j)

def F(coefs, t, y, knots, splrep, proj_matrix, u, lbda, alpha, indices_group):
    """
        This corresponds to the function F(varphi, t) in the paper.
    """
    gradf = grad_func(coefs, y, knots, splrep, proj_matrix, u)
    G_j = len(indices_group)
    z = (coefs - t*gradf)[indices_group]
    S = coordinate_wise_soft_thresholding(z, t * alpha * lbda)
    return max(1 - (t * (1-alpha) * lbda * np.sqrt(G_j))/(l2(S)), 0.0) * S    
        
#%%
def optimize_step_size(coefs, y, knots, splrep, proj_matrix, u, lbda, alpha, indices_group):
    """
        This function aims at finding the optimal step size t.
    """
    t=1.0
    coefs_old = coefs.copy()
    #-------------------------------------------------------------------------------
    resu = F(coefs, t, y, knots, splrep, proj_matrix, u, lbda, alpha, indices_group)
    coefs_new = coefs.copy()
    coefs_new[indices_group] = resu
    #-------------------------------------------------------------------------------
    g = grad_func(coefs_old, y, knots, splrep, proj_matrix, u)
    Delta = coefs_new - coefs_old
    loss_old = loss(coefs_old, y, knots, splrep, proj_matrix, u, 0., 0., 0., 0.)
    loss_new = loss(coefs_new, y, knots, splrep, proj_matrix, u, 0., 0., 0., 0.)
    while loss_new > loss_old + np.dot(Delta,g) + 1/(2*t)*l2(Delta)**2:
        t = t*0.8
        #-------------------------------------------------------------------------------
        resu = F(coefs, t, y, knots, splrep, proj_matrix, u, lbda, alpha, indices_group)
        coefs_new = coefs.copy()
        coefs_new[indices_group] = resu
        #-------------------------------------------------------------------------------
        Delta = coefs_new - coefs_old
        loss_new = loss(coefs_new, y, knots, splrep, proj_matrix, u, 0., 0., 0., 0.)
    return t

def block_wise_descent_fitting(coefs, y, knots, splrep, proj_matrix, u, lbda, alpha, indices_group, rtol=1e-6, max_iter=1000):
    """
        This method is meant to implement the block wise descent algorithm. 
    """    
    l = 1
    cond = 1.0 # cond can be any value which is larger than rtol 
    mu_old = coefs.copy()
    tmp_mu = mu_old
    tmp_coefs = coefs.copy()
    while cond > rtol and l <= max_iter:
        mu_old = tmp_mu
        coefs_old = tmp_coefs
        t = optimize_step_size(coefs_old, y, knots, splrep, proj_matrix, u, lbda, alpha, indices_group)
        #-------------------------------------------------------------------------------
        resu = F(coefs_old, t, y, knots, splrep, proj_matrix, u, lbda, alpha, indices_group)
        mu_new = mu_old
        mu_new[indices_group] = resu
        #-------------------------------------------------------------------------------
        coefs_new = mu_new + l/(l+3)*(mu_new - mu_old)
        tmp_mu = mu_new
        tmp_coefs = coefs_new
        l = l+1
        cond = l2(coefs_new - coefs_old)
    return coefs_new[indices_group]
#%%

def get_coef_exclude_ind(coefs, ind):
    return np.array(list(coefs[:ind]) + list(coefs[ind+1:]))

def get_coef_include_ind(coefs, value, ind):
    tmp = list(coefs)
    tmp.insert(ind, value)
    return np.array(tmp)

def obj_func(x, coef_k, ind, y, knots, splrep, proj_matrix, u, lbda1, alpha1, lbda2, alpha2):
    coefs = get_coef_include_ind(coef_k, x, ind)
    return loss(coefs, y, knots, splrep, proj_matrix, u, lbda1, alpha1, lbda2, alpha2)
    
def partial_f_beta0(coefs, y, knots, splrep, proj_matrix,u):
    coefs = np.asarray(coefs)
    knots = np.asarray(knots)
    coefs = coefs.reshape((2,int(len(coefs)/2)))
    betas = coefs[0] # First half of coefs is bates
    thetas = coefs[1] # Second half of coefs is thetas
    K, p = knots.shape
    y = np.asarray(y)
    y = y-u

    kappa = get_regfunc(betas, splrep, proj_matrix)
    tau = get_regfunc(thetas, splrep, proj_matrix)
    # =========================================================================
    # partial_F/ partial_beta_0
    dF_dbeta1 = np.exp(-kappa)*np.log(1+np.exp(kappa-tau)*np.maximum(y,0.0))-(1+np.exp(-kappa))*(np.exp(kappa-tau)*np.maximum(y,0))/(1+np.exp(kappa-tau)*np.maximum(y,0.0))
    dF_dbeta1[np.where(y < 0.0)] = 0.0
    return -np.mean(dF_dbeta1)

def gradient_f_beta0(x0, coef_k, ind, y, knots, splrep, proj_matrix, u):
    """
        This method returns the absolute value of the partial derivative of Kullback-Leibler divergence with respect to beta_0. 
    """
    x0 = float(x0)
    coefs = get_coef_include_ind(coef_k, x0, ind)
    return abs(partial_f_beta0(coefs, y, knots, splrep, proj_matrix, u))

def partial_f_theta0(coefs, y, knots, splrep, proj_matrix,u):
    coefs = np.asarray(coefs)
    knots = np.asarray(knots)
    coefs = coefs.reshape((2,int(len(coefs)/2)))
    betas = coefs[0] # First half of coefs is bates
    thetas = coefs[1] # Second half of coefs is thetas
    K, p = knots.shape
    y = np.asarray(y)
    y = y-u

    kappa = get_regfunc(betas, splrep, proj_matrix)
    tau = get_regfunc(thetas, splrep, proj_matrix)
    # =========================================================================
    # partial_F/ partial_theta0
    dF_dtheta1 = -1 + (1+np.exp(-kappa))*(np.exp(kappa-tau)*np.maximum(y,0.0))/(1+np.exp(kappa-tau)*np.maximum(y,0.0))
    dF_dtheta1[np.where(y < 0.0)] = 0.0
    return -np.mean(dF_dtheta1)

def gradient_f_theta0(x0, coef_k, ind, y, knots, splrep, proj_matrix, u):
    """
        This method returns the absolute value of the partial derivative of Kullback-Leibler divergence with respect to theta_0.
    """
    x0 = float(x0)
    coefs = get_coef_include_ind(coef_k, x0, ind)
    return abs(partial_f_theta0(coefs, y, knots, splrep, proj_matrix, u))

    