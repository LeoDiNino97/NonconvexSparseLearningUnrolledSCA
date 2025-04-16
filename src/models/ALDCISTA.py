import numpy as np
import cvxpy as cp

import torch
import torch.nn.functional as F
import torch.nn as nn


import warnings
warnings.filterwarnings("ignore")

class ALDC_ISTA(nn.Module):
    def __init__(
            self, 
            A, 
            mode, 
            lambd=0.1, 
            p =0.012, 
            p_max = 0.12, 
            T = 5, 
            W = None,
            SS = False):
        
        super().__init__()
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert mode in ['EXP', 'PNEG', 'SCAD', 'MCP']

        self.mode = mode
        self.linear_shared = True
        self.A = A.to(self.device)  # Move A to the correct device
        self.A.requires_grad = False

        self.T = T

        self.L = torch.max(torch.real(torch.linalg.eigvals(A.t() @ A))).to(self.device)

        # Initialization of the learnable parameters

        #self.lambd = [torch.tensor(lambd / self.L, device=self.device).reshape(1, 1) for _ in range(self.T)]
        
        self.lambd = nn.ParameterList([
            nn.Parameter(torch.tensor(1).reshape(1, 1).to(self.device) * lambd / self.L, requires_grad=True)
            for _ in range(self.T)
        ])
        
        self.mu = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0).reshape(1, 1).to(self.device), requires_grad=True)
            for _ in range(self.T)
        ])

        self.theta = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0).reshape(1, 1).to(self.device), requires_grad=True)
            for _ in range(self.T)
            ])
        
        # Linear layer
        if W is None:
            self.W = self.W_optimization().to(self.device)
        else:
            self.W = W.to(self.device)
            
        self.W1 = torch.clone((self.W.T @ self.A)).to(self.device)
        self.W2 = torch.clone(self.W.T).to(self.device)

        # Support selection mechanism parameters
        self.p = p
        self.p_max = p_max
        if SS: 
            self._shrink = self._shrink_SS
        else:
            self._shrink = self._shrink_FS

        # Losses when doing inference (placeholder for NMSE accumulation)
        self.losses = torch.zeros(self.T, device=self.device)
        self.est_powers = torch.zeros(self.T, device=self.device)

        # Mode setting for the layer
        if mode == 'EXP':

            self.ddx = self.ddxEXP
            self.eta = self.etaEXP

        if mode == 'PNEG':

            self.P = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0).reshape(1, 1).to(self.device) * (-0.1), requires_grad=True)
            for _ in range(self.T)
            ])
            self.ddx = self.ddxNEG
            self.eta = self.etaNEG

        if mode == 'SCAD':
       
            self.a = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0).reshape(1, 1).to(self.device), requires_grad=True)
            for _ in range(self.T)
            ])        
            self.ddx = self.ddxSCAD
            self.eta = self.etaSCAD

        if mode == 'MCP':

            self.ddx = self.ddxMCP
            self.eta = self.etaMCP
    #_________________________________________________________________
    #______________ METHODS FOR ENABLING DIFFERENT LAYERS_____________
    #_________________________________________________________________

    def W_optimization(self):
        N, M = self.A.shape
        W = cp.Variable((N, M))

        objective = cp.Minimize(cp.norm(W.T @ self.A.cpu().numpy(), 'fro'))
        constraints = [W[:, m].T @ self.A.cpu().numpy()[:, m] == 1 for m in range(M)]
        prob = cp.Problem(objective, constraints)

        prob.solve(solver=cp.MOSEK)
        print('Linear layer initialized minimizing coherence!')
        return torch.from_numpy(W.value).float()
    
    # Derivatives of the nonconvex components
    def ddxEXP(self, x, t):
        return torch.sign(x) * self.theta[t] * (1 - torch.exp(-self.theta[t] * torch.abs(x)))

    def ddxNEG(self, x, t):
        return -torch.sign(x) * self.P[t] * self.theta[t] * (1 - (1 + self.theta[t] * torch.abs(x)) ** (self.P[t] - 1))

    def ddxSCAD(self, x, t):
        abs_x = torch.abs(x)

        mask1 = (abs_x <= 1)
        mask2 = (1 / self.theta[t] < abs_x) & (abs_x <= (2 + F.softplus(self.a[t])) / self.theta[t])
        mask3 = (abs_x > (2 + F.softplus(self.a[t])) / self.theta[t])

        # Compute the value for each condition
        val1 = torch.zeros_like(x)
        val2 = torch.sign(x) * (2 * self.theta[t] * (self.theta[t] * abs_x - 1)) / ((2 + F.softplus(self.a[t])) ** 2 - 1)
        val3 = torch.sign(x) * (2 * self.theta[t] / ((2 + F.softplus(self.a[t])) + 1))

        # Apply the masks to compute the final result
        result = torch.where(mask1, val1, torch.where(mask2, val2, val3))

        return result
    
    def ddxMCP(self, x, t):

        abs_x = torch.abs(x)
        mask1 = (abs_x < 1 )
        mask2 = (abs_x >= 1)
        
        theta = 1 / (4 * self.lambd[t])
        val1 = 1 / (2 * theta) * x
        val2 = 1 / (2 * theta) * torch.sign(x)
        
        result = torch.where(mask1, val1, val2)
        return result
    
    # Parametrization for the surrogates
    def etaEXP(self, t):
        return self.theta[t]

    def etaNEG(self, t):
        return -self.P[t] * self.theta[t]

    def etaSCAD(self, t):
        return 2 * self.theta[t] / ((2 + F.softplus(self.a[t])) + 1)
    
    def etaMCP(self, t):
        return 1 
    
    #___________________________________________________________________
    #___________________________________________________________________
    
    def _shrink_FS(self, x, beta, t):
        # Apply soft thresholding directly to all elements
        return beta * F.softshrink(x / beta, lambd=1)

    def _shrink_SS(self, x, beta, t):
        # Get the absolute values of the elements in x
        abs_x = torch.abs(x)
        
        # Sort the elements of x by magnitude along the last dimension (num_features)
        sorted_abs_x, _ = torch.sort(abs_x, dim=-1, descending=True)

        # Determine the threshold index corresponding to the top p% elements in each sample
        p = torch.min(torch.tensor([self.p * (t + 1), self.p_max], device=self.device))
        threshold_idx = int(p * x.shape[-1])
        
        # Get the magnitude threshold for the top p% of elements (per batch)
        if threshold_idx > 0:
            threshold_value = sorted_abs_x[:, threshold_idx - 1:threshold_idx]  # Shape: (batch_size, 1)
        else:
            threshold_value = torch.zeros(x.shape[0], 1, device=x.device)  # Shape: (batch_size, 1)

        # Create a mask to exclude the top p% of elements from shrinkage
        mask = abs_x >= threshold_value
        
        # Apply soft thresholding only to elements outside the top p%
        x_shrink = beta * F.softshrink(x / beta, lambd=1)
        
        # Return the original values for the top p% and the shrinked values for others
        return torch.where(mask, x, x_shrink)

    
    def forward(self, y, its=None, S=None):     
        # Move inputs to the correct device
        if its is None:
            its = self.T - 1
            
        y = y.to(self.device)
        if S is not None:
            S = S.to(self.device)

        res = torch.matmul(y, self.W2.t())
        # Loop over layers to refine the estimate
        for t in range(0, its + 1):
            if t == 0:
                x = self._shrink(self.mu[0] * res , self.eta(0) * self.lambd[0], 0)
            else:
                x = self._shrink(x - self.mu[t] * (torch.matmul(x, self.W1.t()) - res) + self.lambd[t] * self.ddx(x, t), 
                                self.eta(t) * self.lambd[t],
                                t)
                
            if S is not None:
                self.update_loss_metrics(t, x, S)

        return x

    def update_loss_metrics(self, t, x, S):
        with torch.no_grad():
            mse_loss = F.mse_loss(x.detach(), S.detach(), reduction="sum")
            signal_power = torch.sum(S.detach() ** 2)

            self.losses[t] += mse_loss.item()
            self.est_powers[t] += signal_power.item() + 1e-6

    def compute_nmse_inference(self, test_loader):
        # Reset the losses accumulator
        self.losses = torch.zeros(self.T, device=self.device)
        self.est_powers = torch.zeros(self.T, device=self.device)
        
        # Iterate over test_loader
        for _, (Y, S) in enumerate(test_loader):
            Y, S = Y.to(self.device), S.to(self.device)
            _ = self.forward(y=Y, its=None, S=S)  # This will accumulate NMSE values
        
        # Convert accumulated NMSE to dB
        nmse_db = 10 * torch.log10(self.losses / self.est_powers)
        
        # Reset the losses after inference
        self.losses = torch.zeros(self.T, device=self.device)
        self.est_powers = torch.zeros(self.T, device=self.device)

        # Return NMSE in dB for each layer
        return nmse_db


