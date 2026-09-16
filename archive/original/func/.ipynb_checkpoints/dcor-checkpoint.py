"""Computes the distance correlation between two matrices.
https://en.wikipedia.org/wiki/Distance_correlation
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch



def dcov(X, Y):
    """Computes the distance covariance between matrices X and Y.
    """
    n = X.shape[0]
    XY = np.multiply(X, Y)
    cov = np.sqrt(XY.sum()) / n
    return cov


def dvar(X):
    """Computes the distance variance of a matrix X.
    """
    return np.sqrt(np.sum(X ** 2 / X.shape[0] ** 2))


def cent_dist(X):
    """Computes the pairwise euclidean distance between rows of X and centers
     each cell of the distance matrix with row mean, column mean, and grand mean.
    """
    M = squareform(pdist(X))    # distance matrix
    rmean = M.mean(axis=1)
    cmean = M.mean(axis=0)
    gmean = rmean.mean()
    R = np.tile(rmean, (M.shape[0], 1)).transpose()
    C = np.tile(cmean, (M.shape[1], 1))
    G = np.tile(gmean, M.shape)
    CM = M - R - C + G
    return CM


def compute_dcor(X, Y):
    """Computes the distance correlation between two matrices X and Y.
    X and Y must have the same number of rows.
    >>> X = np.matrix('1;2;3;4;5')
    >>> Y = np.matrix('1;2;9;4;4')
    >>> dcor(X, Y)
    0.76267624241686649
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    assert X.shape[0] == Y.shape[0]

    A = cent_dist(X)
    B = cent_dist(Y)

    dcov_AB = dcov(A, B)
    dvar_A = dvar(A)
    dvar_B = dvar(B)

    dcor = 0.0
    if dvar_A > 0.0 and dvar_B > 0.0:
        dcor = dcov_AB / np.sqrt(dvar_A * dvar_B)

    return dcor


def calculate_distance_matrix(X):
    """
    Calculate the Euclidean distance matrix for a set of vectors.
    
    Parameters:
    vectors (numpy.ndarray): Array of vectors of shape (n, d) where n is the number of vectors
                            and d is the dimension of each vector.
    
    Returns:
    numpy.ndarray: Distance matrix of shape (n, n).
    """
    distance_matrix = squareform(pdist(X)) 
    return distance_matrix

def calculate_c_matrix(distance_matrix):
    """
    Calculate the C matrix as defined in the UDcorr formula.
    
    Parameters:
    distance_matrix (numpy.ndarray): Distance matrix of shape (n, n).
    
    Returns:
    numpy.ndarray: C matrix of shape (n, n).
    """
    n = len(distance_matrix)
    c_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Term 1: D_ij
                term1 = distance_matrix[i, j]
                
                # Term 2: 1/(n-2) ∑(t=1 to n) D_it
                term2 = np.sum(distance_matrix[i, :]) / (n)
                
                # Term 3: 1/(n-2) ∑(s=1 to n) D_sj
                term3 = np.sum(distance_matrix[:, j]) / (n)
                
                # Term 4: 1/((n-1)(n-2)) ∑(s,t=1 to n) D_st
                term4 = np.sum(distance_matrix) / ((n) * (n))
                
                c_matrix[i, j] = term1 - term2 - term3 + term4
    
    return c_matrix

def calculate_udcov(c_matrix_x, c_matrix_y):
    """
    Calculate the Unbiased Distance Covariance (UDcov).
    
    Parameters:
    c_matrix_x (numpy.ndarray): C matrix for X of shape (n, n).
    c_matrix_y (numpy.ndarray): C matrix for Y of shape (n, n).
    
    Returns:
    float: The unbiased distance covariance.
    """
    n = len(c_matrix_x)
    
    # Calculate trace(C^x C^y) - this is the sum of element-wise products
    trace_sum = np.trace(c_matrix_x @ c_matrix_y)
    
    return trace_sum / (n * (n))

def compute_udcorr(X, Y):
    """
    Calculate the Unbiased Distance Correlation (UDcorr).
    
    Parameters:
    x (numpy.ndarray): Array of vectors X of shape (n, d_x).
    y (numpy.ndarray): Array of vectors Y of shape (n, d_y).
    
    Returns:
    float: The unbiased distance correlation.
    """
    if isinstance(X, torch.Tensor):
        x = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        y = Y.detach().cpu().numpy()

    x = np.array(x)
    y = np.array(y)
    
    # Calculate distance matrices
    distance_matrix_x = calculate_distance_matrix(x)
    distance_matrix_y = calculate_distance_matrix(y)
    
    # Calculate C matrices
    c_matrix_x = calculate_c_matrix(distance_matrix_x)
    c_matrix_y = calculate_c_matrix(distance_matrix_y)
    
    # Calculate UDcov values
    udcov_xy = calculate_udcov(c_matrix_x, c_matrix_y)
    udcov_xx = calculate_udcov(c_matrix_x, c_matrix_x)
    udcov_yy = calculate_udcov(c_matrix_y, c_matrix_y)
    
    # Handle potential division by zero
    if udcov_xx <= 0 or udcov_yy <= 0:
        return 0
    
    # Calculate UDcorr
    return udcov_xy / np.sqrt(udcov_xx * udcov_yy)

