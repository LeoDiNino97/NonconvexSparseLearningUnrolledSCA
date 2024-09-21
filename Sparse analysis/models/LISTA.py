import torch
import torch.nn.functional as F
import torch.nn as nn

class LISTA_Net(nn.Module):
    def __init__(self, A, beta_ = 0.1, K = 5):
        super(LISTA_Net, self).__init__()

        # Number of layers <-> iterations
        self.K = K

        # Parameters
        self.W1 = nn.Linear(A.shape[1], A.shape[0], bias=False)
        self.W2 = nn.Linear(A.shape[1], A.shape[1], bias=False)

        norm = (1.001*torch.linalg.norm(A.T @ A, 2))
        self.beta = nn.Parameter(torch.ones(self.K + 1, 1, 1) * beta_ / norm, requires_grad=True)       
        self.mu = nn.Parameter(torch.ones(self.K + 1, 1, 1) / norm, requires_grad=True)

        self.W1.weight.data = A.t() 
        self.W2.weight.data = A.t() @ A 

        # Losses when doing inference (placeholder for NMSE accumulation)
        self.losses = torch.zeros(self.K)
        self.est_powers = torch.zeros(self.K)

    def _shrink(self, x, beta):
        return beta * F.softshrink(x / beta, lambd=1)
    
    def forward(self, y, S=None):     
        # Initial sparse estimate using the first layer
        x = self._shrink(self.mu[0,:,:] * self.W1(y), self.beta[0, : ,:])
        
        # Loop over layers to refine the estimate
        for i in range(1, self.K + 1):
            x = self._shrink(x - self.mu[i, :, :] * (self.W2(x) - self.W1(y)), self.beta[i, : ,:])
            
            if S is not None:  # During inference, compute the NMSE at each layer
                with torch.no_grad():
                    mse_loss = F.mse_loss(x.detach(), S.detach(), reduction="sum")
                    signal_power = torch.sum(S.detach() ** 2)

                    self.losses[i - 1] += mse_loss.item()
                    self.est_powers[i - 1] += signal_power.item() + 1e-6

        return x

    # Method to compute NMSE during inference mode
    def compute_nmse_inference(self, test_loader):
        # Reset the losses accumulator
        self.losses = torch.zeros(self.K)
        
        # Iterate over test_loader
        for _, (Y, S) in enumerate(test_loader):
            _ = self.forward(Y, S)  # This will accumulate NMSE values
        
        # Convert accumulated NMSE to dB
        nmse_db = 10 * torch.log10(self.losses / self.est_powers)
        
        # Reset the losses after inference
        self.losses = torch.zeros(self.K)
        self.est_powers = torch.zeros(self.K)

        # Return NMSE in dB for each layer
        return nmse_db.cpu().numpy()  # Convert to NumPy array for ease of use