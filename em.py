"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture

pi = np.pi
#if you come back to this and vectorize it and you want to have just a logged gauss function you can just remove p[j]
#this may help with code modulation
def logged_gauss_incomplete_p_non_vect(X, mu, var, p):
    n = X.shape[0]
    k = mu.shape[0]
    output_mat = np.zeros([n,k])
    for i in range(n):
        indices = np.nonzero(X[i])[0]
        d = len(indices)
        for j in range(k):
            summ = 0
            for index in indices:
                summ += (X[i][index] - mu[j][index])**2
            
            exp_factor_logged = -summ/(2*var[j])
            first_factor_logged = np.log(p[j]) - (d/2)*np.log(2*pi*var[j])
            output_mat[i][j] = first_factor_logged + exp_factor_logged
            
    
    return output_mat

def logged_gauss_incomplete_p(X, mu, var, p):
    
    n,d = X.shape
    k = var.shape[0]
    delta = np.where(X == 0, 0, 1) #nxd
    delta_reshaped = delta.reshape([delta.shape[0],1,delta.shape[1]]) #nx1xd
    X_reshaped = X.reshape([n,1,d])
    u_3d = (mu*np.ones([n,k,d]))*delta_reshaped
    sub_stack = u_3d-X_reshaped #nxkxd
    norm_squared = np.sum(sub_stack*sub_stack, axis = 2)#nxk
    
    exp_factor_logged = -norm_squared/(2*var)
    C_u = np.sum(delta, axis = 1, keepdims=True)
    
    var_2d = var*np.ones([n,k])
    first_factor_logged = np.log(p) - (C_u/2)*np.log(2*pi*var_2d)
    
    return first_factor_logged + exp_factor_logged
    


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    mu, var, p = mixture
    logged_gauss_p = logged_gauss_incomplete_p(X,mu,var,p)
    max_vector = np.amax(logged_gauss_p, axis=1, keepdims=True)
    scaled_gauss = np.exp(logged_gauss_p - max_vector)
    denominator_logged = max_vector + np.log(np.sum(scaled_gauss, axis = 1, keepdims=True))
    log_post = logged_gauss_p - denominator_logged
    log_likelihood = np.sum(denominator_logged)
    
    return np.exp(log_post), log_likelihood


def mstep_non_vect(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    mu, var, p = mixture
    
    n, d = X.shape
    k = post.shape[1]
    new_p = np.sum(post , axis = 0)/n
    
    delta = np.where(X == 0, 0, 1) #nxd
    mu_numerator = np.dot(X.T, post).T
    mu_denominator = np.dot(delta.T, post).T
    new_mu = np.where(mu_denominator >= 1, mu_numerator/mu_denominator, mu)
    
    #computing normed_squared, not vectorized yet
    norm_squared = np.zeros([n,k]) #nxk
    for i in range(n):
        for j in range(k):
            indices = np.nonzero(X[i])[0]
            d = len(indices)
            summ = 0
            for index in indices:
                summ += (X[i][index] - new_mu[j][index])**2 
            norm_squared[i][j] = summ
            
    C_u = np.sum(delta, axis = 1, keepdims=True)
    
    summation_factor = np.sum(post*norm_squared, axis = 0)
    first_factor = 1/np.sum(C_u*post, axis = 0)
    var_bad = first_factor*summation_factor
    new_var = np.where(var_bad < min_variance, min_variance, var_bad)
    
    mixture.mu[:] = new_mu[:]
    mixture.var[:] = new_var[:]
    mixture.p[:] = new_p[:]
    
    return mixture
    
def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    mu, var, p = mixture
    n, d = X.shape
    k = post.shape[1]
    new_p = np.sum(post , axis = 0)/n
    
    delta = np.where(X == 0, 0, 1) #nxd
    mu_numerator = np.dot(X.T, post).T
    mu_denominator = np.dot(delta.T, post).T
    new_mu = np.where(mu_denominator >= 1, mu_numerator/(mu_denominator + 1e-16), mu) #kxd
    
    #vectorized computing norm_squared (I think).
    delta_reshaped = delta.reshape([delta.shape[0],1,delta.shape[1]]) #nx1xd
    X_reshaped = X.reshape([n,1,d])
    u_3d = (new_mu*np.ones([n,k,d]))*delta_reshaped
    sub_stack = u_3d-X_reshaped #nxkxd
    norm_squared = np.sum(sub_stack*sub_stack, axis = 2)#nxk
            
    C_u = np.sum(delta, axis = 1, keepdims=True)
    
    summation_factor = np.sum(post*norm_squared, axis = 0)
    first_factor = 1/np.sum(C_u*post, axis = 0)
    var_bad = first_factor*summation_factor
    new_var = np.where(var_bad < min_variance, min_variance, var_bad)
    
    mixture.mu[:] = new_mu[:]
    mixture.var[:] = new_var[:]
    mixture.p[:] = new_p[:]
    
    return mixture


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    flag = False
    while True:
        post, new_log_likelihood = estep(X, mixture)
        if flag and new_log_likelihood - old_log_likelihood <= abs(new_log_likelihood)/(10**6):
            return mixture, new_log_likelihood, post
        old_log_likelihood = new_log_likelihood
        mixture = mstep(X, post, mixture)
        flag = True

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    mu = mixture[0]
    post = estep(X, mixture)[0]
    new_values = np.dot(post, mu)
    new_X = np.where(X==0, new_values, X)
    
    return new_X
    
    
    
    
