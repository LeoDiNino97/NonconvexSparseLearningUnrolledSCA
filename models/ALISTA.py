import cvxpy as cp
import torch
import torch.nn.functional as F
import torch.nn as nn

class ALISTA(nn.Module):
    def __init__(self, A, beta_ = 0.1, T=5, p = 0.012, p_max = 0.12):
        super(ALISTA, self).__init__()

        # Set device (CPU or GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Number of layers <-> iterations
        self.T = T
        self.linear_shared = True

        # Parameters
        self.A = A.to(self.device)
        self.W = self.W_optimization().to(self.device)

        norm = (1.001 * torch.linalg.norm(self.A.T @ self.A, 2))
        self.beta = nn.ParameterList([
            nn.Parameter(torch.tensor(beta_ / norm).reshape(1, 1).to(self.device), requires_grad=True)
            for _ in range(self.T + 1)
        ])

        self.mu = nn.ParameterList([
            nn.Parameter(torch.tensor(1 / norm).reshape(1, 1).to(self.device), requires_grad=True)
            for _ in range(self.T + 1)
        ])

        self.W1 = torch.clone((self.W.T @ self.A)).to(self.device)
        self.W2 = torch.clone(self.W.T).to(self.device)
        
        # Support selection mechanism parameters
        self.p = p
        self.p_max = p_max

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
        if its is None:
            its = self.T
            
        y = y.to(self.device)
        if S is not None:
            S = S.to(self.device)

        # Initial estimation with shrinkage
        h = self.mu[0] * torch.matmul(y, self.W2.t())
        x = self._shrink(h, self.beta[0], 1)
        
        for t in range(1, its + 1):
            k = self.mu[t] * (torch.matmul(x, self.W1.t()) - torch.matmul(y, self.W2.t()))
            h = x - k
            x = self._shrink(h, self.beta[t], t)

            # If ground truth is provided, calculate the loss for monitoring
            if S is not None:
                with torch.no_grad():
                    mse_loss = F.mse_loss(x.detach(), S.detach(), reduction="sum")
                    signal_power = torch.sum(S.detach() ** 2)

                    self.losses[t - 1] += mse_loss.item()
                    self.est_powers[t - 1] += signal_power.item() + 1e-6

        return x
    
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
        
    def compute_support(self, test_loader):
        # Reset the losses accumulator
        self.losses = torch.zeros(self.T, device=self.device)
        
        total_precision = 0.0
        total_samples = 0
        
        # Iterate over test_loader
        for _, (Y, S) in enumerate(test_loader):
            Y, S = Y.to(self.device), S.to(self.device)
            X = self.forward(y = Y, its = None)

            # Hard threshold X by retaining the top `supp` largest components in absolute value
            X_thresholded = (X != 0).float()  # Create a binary support mask for X

            # Create a binary support mask for S (ground truth support)
            S_support = (S != 0).float()

            # True positives (correctly identified non-zeros)
            true_positives = (X_thresholded * S_support).sum(dim=1)

            # Predicted positives (all non-zeros in X_thresholded)
            predicted_positives = X_thresholded.sum(dim=1)

            # Precision = True positives / Predicted positives (avoiding division by zero)
            precision = true_positives / (predicted_positives + 1e-10)

            # Sum precision for this batch
            total_precision += precision.sum().item()
            total_samples += Y.size(0)
        
        # Compute the average precision over all batches
        average_precision = total_precision / total_samples
        return average_precision
