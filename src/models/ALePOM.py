import cvxpy as cp
import torch
import torch.nn.functional as F
import torch.nn as nn

class ALePOM(nn.Module):
    def __init__(
            self, 
            A, 
            beta_ = 0.1, 
            T=5,
            W = None):
        
        super(ALePOM, self).__init__()

        # Set device (CPU or GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Number of layers <-> iterations
        self.T = T
        self.linear_shared = True

        # Parameters
        self.A = A.to(self.device)

        # Linear layer
        if W is None:
            self.W = self.W_optimization().to(self.device)
        else:
            self.W = W.to(self.device)
    
        norm = (1.001 * torch.linalg.norm(self.A.T @ self.A, 2))
        self.beta = nn.ParameterList([
            nn.Parameter(torch.tensor(beta_ / norm).reshape(1, 1).to(self.device), requires_grad=True)
            for _ in range(self.T + 1)
        ])

        self.mu = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0).reshape(1, 1).to(self.device), requires_grad=True)
            for _ in range(self.T + 1)
        ])

        self.W1 = torch.clone((self.W.T @ self.A)).to(self.device)
        self.W2 = torch.clone(self.W.T).to(self.device)

        # Losses when doing inference
        self.losses = torch.zeros(self.T, device=self.device)
        self.est_powers = torch.zeros(self.T, device=self.device)

    def W_optimization(self):
        N, M = self.A.shape
        W = cp.Variable((N, M))

        objective = cp.Minimize(cp.norm(W.T @ self.A.cpu().numpy(), 'fro'))
        constraints = [W[:, m].T @ self.A.cpu().numpy()[:, m] == 1 for m in range(M)]
        prob = cp.Problem(objective, constraints)

        prob.solve(solver=cp.MOSEK)
        print('Linear layer initialized minimizing coherence!')
        return torch.from_numpy(W.value).float()

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
    
    def forward(self, y, its=None, S=None):
        if its is None:
            its = self.T          
        y = y.to(self.device)
        if S is not None:
            S = S.to(self.device)

        res = torch.matmul(y, self.W2.t())
        x = None
        for t in range(0, its + 1):
            if t == 0:
                h = self.mu[0] * res
            else:
                k = self.mu[t] * (torch.matmul(x, self.W1.t()) - res)
                h = x - k

            x = self.proximal(h, self.beta[t])

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
        self.losses = torch.zeros(self.T + 1, device=self.device)
        self.est_powers = torch.zeros(self.T + 1, device=self.device)
        
        # Iterate over test_loader
        for _, (Y, S) in enumerate(test_loader):
            Y, S = Y.to(self.device), S.to(self.device)
            _ = self.forward(y=Y, its=None, S=S)  # This will accumulate NMSE values
        
        # Convert accumulated NMSE to dB
        nmse_db = 10 * torch.log10(self.losses / self.est_powers)
    

        # Return NMSE in dB for each layer
        return nmse_db

