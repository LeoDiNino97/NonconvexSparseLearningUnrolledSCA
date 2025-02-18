import torch
import torch.nn.functional as F
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

class LePOM_CP(nn.Module):
    def __init__(self, A, beta_ = 0.1, T = 5):
        super(LePOM_CP, self).__init__()

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
        self.Ws = nn.ModuleList()
        for _ in range(self.T + 1):
            W = nn.Linear(A.shape[1], A.shape[0], bias=False).to(self.device)
            W.weight.data = torch.clone(self.A.T).to(self.device) / norm
            self.Ws.append(W)

        # Losses when doing inference
        self.losses = torch.zeros(self.T, device=self.device)
        self.est_powers = torch.zeros(self.T, device=self.device)

    def proximal(self, x, beta):
        # Apply soft thresholding directly to all elements
        abs_x_tilde = torch.abs(x)
        sign_x_tilde = torch.sign(x)

        eta = 1 / (4 * beta)
        # Compute the conditions
        cond1 = abs_x_tilde >= (1 / (2 * eta))
        cond2 = (beta <= abs_x_tilde) & (abs_x_tilde < (1 / (2 * eta)))
        cond3 = abs_x_tilde < beta

        # Compute updates based on conditions
        case1 = x
        case2 = (x - beta * sign_x_tilde) / (1 - 2 * beta * eta)
        case3 = torch.zeros_like(x)

        # Apply conditions
        x_next = torch.where(cond1, case1, torch.where(cond2, case2, case3))

        return x_next
    
    def forward(self, y, its = None, S=None):     
        # Move inputs to the correct device

        if its is None:
            its = self.T

        # Move inputs to the correct device
        y = y.to(self.device)
        if S is not None:
            S = S.to(self.device)

        # Initial estimation with shrinkage
        h = self.Ws[0](y)
        x = self.proximal(h, self.beta[0])
        
        for t in range(1, its + 1):
            k = self.Ws[t](torch.matmul(x, self.A.t()) - y)
            h = x - k
            x = self.proximal(h, self.beta[t])

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
