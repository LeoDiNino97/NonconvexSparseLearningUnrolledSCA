import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

class LISTA(nn.Module):
    def __init__(self, A, beta_ = 0.1, T = 5, p = 0.012, p_max = 0.12):
        super(LISTA, self).__init__()

        # Automatically set device to 'cuda' if available, otherwise 'cpu'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Number of layers <-> iterations
        self.T = T
        self.linear_shared = False

        # Parameters
        self.A = A.to(self.device)

        norm = (1.001 * torch.linalg.norm(self.A.T @ self.A, 2))

        self.beta = nn.ParameterList([
            nn.Parameter(torch.tensor(beta_ / norm).reshape(1, 1).to(self.device), requires_grad=True)
            for _ in range(self.T + 1)
        ])
        
        # Linear layers
        self.Ws_1 = nn.ModuleList()
        self.Ws_2 = nn.ModuleList()

        for _ in range(self.T + 1):
            W1 = nn.Linear(A.shape[1], A.shape[0], bias=False).to(self.device)
            W2 = nn.Linear(A.shape[1], A.shape[1], bias=False).to(self.device)

            W1.weight.data = torch.clone(self.A.T).to(self.device) / norm
            W2.weight.data = torch.eye(A.shape[1]).to(self.device) - torch.clone(A.t() @ A).to(self.device)/norm
            self.Ws_1.append(W1)
            self.Ws_2.append(W2)

        # Support selection mechanism parameters
        self.p = p
        self.p_max = p_max

        # Losses when doing inference
        self.losses = torch.zeros(self.T, device=self.device)
        self.est_powers = torch.zeros(self.T, device=self.device)

    def _shrink(self, x, beta, t):
        # Get the absolute values of the elements in x
        abs_x = torch.abs(x)
        
        # Sort the elements of x by magnitude along the last dimension (num_features)
        sorted_abs_x, _ = torch.sort(abs_x, dim=-1, descending=True)

        # Determine the threshold index corresponding to the top p% elements in each sample
        p = torch.min(torch.tensor([self.p * t, self.p_max], device=self.device))
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

    def forward(self, y, its = None, S=None):     
        # Move inputs to the correct device

        if its is None:
            its = self.T
            
        # Move inputs to the correct device
        y = y.to(self.device)
        if S is not None:
            S = S.to(self.device)

        # Initial estimation with shrinkage
        x = torch.zeros((y.shape[0], self.A.shape[1])).to(self.device)
        h = self.Ws_1[0](y) + self.Ws_2[0](x)
        x = self._shrink(h, self.beta[0], 1)
        
        for t in range(1, its + 1):
            x = self._shrink(self.Ws_1[t](y) + self.Ws_2[t](x), self.beta[t], t)

            # If ground truth is provided, calculate the loss for monitoring
            if S is not None:

                with torch.no_grad():

                    mse_loss = F.mse_loss(x.detach(), S.detach(), reduction="sum")
                    signal_power = torch.sum(S.detach() ** 2)

                    self.losses[t - 1] += mse_loss.item()
                    self.est_powers[t - 1] += signal_power.item() + 1e-6
   
        return x

    # Method to compute NMSE during inference mode
    def compute_nmse_inference(self, test_loader):
        # Reset the losses accumulator
        self.losses = torch.zeros(self.T, device=self.device)
        
        # Iterate over test_loader
        for _, (Y, S) in enumerate(test_loader):
            Y, S = Y.to(self.device), S.to(self.device)
            _ = self.forward(y = Y, its = None, S = S)  # This will accumulate NMSE values
        
        # Convert accumulated NMSE to dB
        nmse_db = 10 * torch.log10(self.losses / self.est_powers)
        
        # Reset the losses after inference
        self.losses = torch.zeros(self.T, device=self.device)
        self.est_powers = torch.zeros(self.T, device=self.device)

        # Return NMSE in dB for each layer
        return nmse_db


####################################################
############### RECURRENT DEFINITION ###############
####################################################

class LISTA_Net(nn.Module):

    def __init__(self, A, beta_ = 0.1, T = 5, p = 0.012, p_max = 0.12):
        super(LISTA_Net, self).__init__()

        # Automatically set device to 'cuda' if available, otherwise 'cpu'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Number of layers <-> iterations
        self.T = T
        self.linear_shared = True

        # Move A to device
        A = A.to(self.device)

        # Parameters
        self.Ws_1 = nn.Linear(A.shape[1], A.shape[0], bias=False).to(self.device)
        self.Ws_2 = nn.Linear(A.shape[1], A.shape[1], bias=False).to(self.device)

        norm = (1.001 * torch.linalg.norm(A.T @ A, 2))

        self.beta = nn.ParameterList([
            nn.Parameter(torch.tensor(beta_ / norm).reshape(1, 1).to(self.device), requires_grad=True)
            for _ in range(self.T + 1)
        ])

        self.mu = nn.ParameterList([
            nn.Parameter(torch.tensor(1 / norm).reshape(1, 1).to(self.device), requires_grad=True)
            for _ in range(self.T + 1)
        ])

        self.Ws_1.weight.data = torch.clone(A.t()).to(self.device)
        self.Ws_2.weight.data = torch.clone(A.t() @ A).to(self.device)

        # Support selection mechanism parameters
        self.p = p
        self.p_max = p_max

        # Losses when doing inference (placeholder for NMSE accumulation)
        self.losses = torch.zeros(self.T, device=self.device)
        self.est_powers = torch.zeros(self.T, device=self.device)

    def _shrink(self, x, beta, t):
        # Get the absolute values of the elements in x
        abs_x = torch.abs(x)
        
        # Sort the elements of x by magnitude along the last dimension (num_features)
        sorted_abs_x, _ = torch.sort(abs_x, dim=-1, descending=True)

        # Determine the threshold index corresponding to the top p% elements in each sample
        p = torch.min(torch.tensor([self.p * t, self.p_max], device=self.device))
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

    def forward(self, y, its = None, S=None):     
        # Move inputs to the correct device

        if its is None:
            its = self.T

        y = y.to(self.device)
        if S is not None:
            S = S.to(self.device)

        # Initial sparse estimate using the first layer
        x = self._shrink(self.mu[0] * self.Ws_1(y), self.beta[0], 0)
        
        # Loop over layers to refine the estimate
        for t in range(1, its + 1):
            x = self._shrink(x - self.mu[t] * (self.Ws_2(x) - self.Ws_1(y)), self.beta[t], t)
            
            if S is not None:  # During inference, compute the NMSE at each layer
                with torch.no_grad():
                    mse_loss = F.mse_loss(x.detach(), S.detach(), reduction="sum")
                    signal_power = torch.sum(S.detach() ** 2)

                    self.losses[t - 1] += mse_loss.item()
                    self.est_powers[t - 1] += signal_power.item() + 1e-6

        return x

    # Method to compute NMSE during inference mode
    def compute_nmse_inference(self, test_loader):
        # Reset the losses accumulator
        self.losses = torch.zeros(self.T, device=self.device)
        
        # Iterate over test_loader
        for _, (Y, S) in enumerate(test_loader):
            Y, S = Y.to(self.device), S.to(self.device)
            _ = self.forward(Y, S)  # This will accumulate NMSE values
        
        # Convert accumulated NMSE to dB
        nmse_db = 10 * torch.log10(self.losses / self.est_powers)
        
        # Reset the losses after inference
        self.losses = torch.zeros(self.T, device=self.device)
        self.est_powers = torch.zeros(self.T, device=self.device)

        # Return NMSE in dB for each layer
        return nmse_db
