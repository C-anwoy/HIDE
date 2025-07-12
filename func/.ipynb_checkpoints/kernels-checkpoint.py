import torch
import numpy as np

def rbf_kernel(X, Y=None, gamma=1e-7):
    """Radial Basis Function (RBF) kernel."""
    if Y is None:
        Y = X
    pairwise_sq_dists = torch.cdist(X, Y) ** 2 
    return torch.exp(-gamma * pairwise_sq_dists)

def linear_kernel(X, Y=None):
    """Linear kernel."""
    if Y is None:
        Y = X
    return torch.matmul(X, Y.T)

def polynomial_kernel(X, Y=None, degree=3, coef0=1e-3):
    """Polynomial kernel."""
    if Y is None:
        Y = X
    return (torch.matmul(X, Y.T) + coef0) ** degree

def cosine_kernel(X, Y=None):
    """Cosine similarity kernel."""
    if Y is None:
        Y = X
    X_norm = torch.norm(X, dim=1, keepdim=True)
    Y_norm = torch.norm(Y, dim=1, keepdim=True) if Y is not None else X_norm
    return torch.matmul(X / X_norm, (Y / Y_norm).T)

def sigmoid_kernel(X, Y=None, alpha=1e-1, coef0=1e-1):
    """Sigmoid kernel."""
    if Y is None:
        Y = X
    return torch.tanh(alpha * torch.matmul(X, Y.T) + coef0)

def laplacian_kernel(X, Y=None, gamma=1e-3):
    """Laplacian kernel."""
    if Y is None:
        Y = X
    pairwise_dists = torch.cdist(X, Y, p=1)  # Manhattan distance (p=1)
    return torch.exp(-gamma * pairwise_dists)

def exponential_kernel(X, Y=None, gamma=1e-3):
    """Exponential kernel."""
    if Y is None:
        Y = X
    pairwise_dists = torch.cdist(X, Y)  # Euclidean distance
    return torch.exp(-gamma * pairwise_dists)

def chi2_kernel(X, Y=None, gamma=0.1):
    """Chi-square kernel."""
    if Y is None:
        Y = X
    X = X + 1e-8  # Avoid division by zero
    Y = Y + 1e-8 if Y is not None else X
    pairwise_chi2 = 0.5 * torch.sum((X[:, None, :] - Y[None, :, :]) ** 2 / (X[:, None, :] + Y[None, :, :]), dim=2)
    return torch.exp(-gamma * pairwise_chi2)

def periodic_kernel(X, Y=None, length_scale=40, periodicity=20):
    """Periodic kernel."""
    if Y is None:
        Y = X
    pairwise_dists = torch.cdist(X, Y)
    return torch.exp(-2 * (torch.sin(np.pi * pairwise_dists / periodicity) ** 2) / length_scale ** 2)

def matern_kernel(X, Y=None, nu=0.5, length_scale=64):
    """Matern kernel."""
    if Y is None:
        Y = X
    pairwise_dists = torch.cdist(X, Y)
    if nu == 0.5:
        return torch.exp(-pairwise_dists / length_scale)
    elif nu == 1.5:
        factor = np.sqrt(3) * pairwise_dists / length_scale
        return (1 + factor) * torch.exp(-factor)
    elif nu == 2.5:
        factor = np.sqrt(5) * pairwise_dists / length_scale
        return (1 + factor + (factor ** 2) / 3) * torch.exp(-factor)
    else:
        raise ValueError("Only nu=0.5, 1.5, and 2.5 are supported.")

# Dictionary for easier access to kernels
KERNEL_FUNCTIONS = {
    "rbf": rbf_kernel,
    "linear": linear_kernel,
    "polynomial": polynomial_kernel,
    "cosine": cosine_kernel,
    "sigmoid": sigmoid_kernel,
    "laplacian": laplacian_kernel,
    "exponential": exponential_kernel,
    # "chi2": chi2_kernel,
    "periodic": periodic_kernel,
    "matern": matern_kernel,
}
