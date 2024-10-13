import cvxpy as cp
import torch
import torch.nn.functional as F
import torch.nn as nn

class Precoder(nn.Module):
    def __init__(self, T = 10, ):
        super().__init__()