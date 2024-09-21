import numpy as np
import cvxpy as cp

import scipy

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def ISTA_torch(y, A, S, rho=0.5, L=1, max_itr=300):
    loss = torch.zeros(max_itr)  # Store NMSE in dB
    powers = torch.zeros(max_itr)  # Store NMSE in dB

    x_hat = torch.zeros(A.shape[1])
    proj = torch.nn.Softshrink(lambd= rho / L)

    for idx in range(max_itr):
        # Gradient update step
        x_tilde = x_hat - 1 / L * (A.T @ (A @ x_hat - y))
        x_hat = proj(x_tilde)

        # NMSE computation
        mse_loss = F.mse_loss(x_hat, S, reduction="sum").data.item()
        signal_power = torch.sum(S**2).item()

        loss[idx] = mse_loss
        powers[idx] = signal_power

    return loss, powers

def ISTA_BATCH(test_loader, T, A, rho=1):
    A = A.cpu()
    m = A.shape[1]
    # Find the largest eigenvalue of A^T A (used as L)
    L = float(scipy.linalg.eigvalsh(A.t() @ A, eigvals=(m - 1, m - 1)))

    # Aggregate T iterations' NMSE loss in dB
    losses_ = torch.zeros(T)
    powers_ = torch.zeros(T)

    for _, (y, S) in enumerate(test_loader.dataset):
        losses, powers = ISTA_torch(y=y, A=A, S=S, L=L, max_itr=T)
        losses_ += losses
        powers_ += powers

    # Return the mean NMSE in dB for all batches
    return 10 * torch.log10(losses / powers)


#############################################################

def FISTA_torch(y, A, S, rho=0.5, L=1, max_itr=300):
    losses = torch.zeros(max_itr)  # Store NMSE in dB
    powers = torch.zeros(max_itr)  # Store NMSE in dB
    
    x_hat = torch.zeros(A.shape[1])
    x_old = x_hat.clone()  # Keep track of previous x_hat
    proj = torch.nn.Softshrink(lambd=rho / L)
    t = 1

    for idx in range(max_itr):
        # Gradient update step
        x_tilde = x_hat - 1 / L * (A.T @ (A @ x_hat - y))
        
        # Apply soft-thresholding
        x_new = proj(x_tilde)
        
        # Update momentum parameter
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2

        # Combine the momentum step with the update
        x_hat = x_new + (t - 1) / t_new * (x_new - x_old)

        # NMSE computation
        mse_loss = F.mse_loss(x_hat, S, reduction="sum").data.item()
        signal_power = torch.sum(S**2).item()

        losses[idx] = mse_loss  # NMSE in dB
        powers[idx] = signal_power

        # Update for the next iteration
        x_old = x_new.clone()
        t = t_new

    return losses, powers


def FISTA_BATCH(test_loader, T, A, rho=1):
    A = A.cpu()
    m = A.shape[1]
    # Find the largest eigenvalue of A^T A (used as L)
    L = float(scipy.linalg.eigvalsh(A.t() @ A, eigvals=(m - 1, m - 1)))

    # Aggregate T iterations' NMSE loss in dB
    losses_ = torch.zeros(T)
    powers_ = torch.zeros(T)
    for idx, (y, S) in enumerate(test_loader.dataset):
        losses, powers = FISTA_torch(y=y, A=A, S=S, L=L, max_itr=T)
        losses_ += losses
        powers_ += powers

    # Return the mean NMSE in dB for all batches
    return 10 * torch.log10(losses / powers)