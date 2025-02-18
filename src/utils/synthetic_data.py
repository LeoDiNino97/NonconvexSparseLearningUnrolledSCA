import numpy as np
import torch
import torch.utils.data as Data

class SyntheticSignals():
    def __init__(self, A, n, m, p=0.1, SNR=None, size=1000, batch_size=512, discretized=False):
        
        # Model complexity
        self.n = n  # Number of samples in the original signal
        self.m = m  # Number of samples through the linear transformation

        # Sensing matrix
        if A is None:                            
            self.A = self.A_initialization()          
        else:
            assert (A.shape[0] == m and A.shape[1] == n)
            self.A = A

        # Sparsity and noise
        self.p = p  # Sparsity of the signal in terms of percentage of components being non-zero
        self.SNR = SNR  # Signal-to-noise ratio 
        self.discretized = discretized

        # Size and batch size
        self.batch_size = batch_size
        self.size = size  # Size of the dataset

        # Preallocation
        self.x = torch.zeros(self.size, self.n)
        self.y = torch.zeros(self.size, self.m)
        
        # Generating the dataset
        self.set_data()

    def A_initialization(self):
        A = torch.normal(0, torch.sqrt(torch.tensor(1/self.m)), size=(self.m, self.n))
        A /= torch.linalg.norm(A, dim=0)  # Normalize columns
        return A
    
    def set_tuple(self, i):
        # Reset the signal to zero before generating a new sparse signal
        self.x[i, :] = 0

        # Generating random sparsity in the canonical basis of the original signal
        idxs = np.where(np.random.rand(self.n) < self.p)[0]  
        if self.discretized:
          peaks = np.random.choice([-1, 1], size=idxs.shape[0])
        else:
          peaks = np.random.normal(loc = 0, scale=1, size=idxs.shape[0])

        # Generating the original signal and its corrupted observations
        self.x[i, idxs] = torch.from_numpy(peaks).to(self.x)
        self.y[i, :] = self.A @ self.x[i, :] 
        
        # Adding noise based on the SNR if provided
        if self.SNR is not None:
            var = torch.mean(self.y[i, :]**2) / self.SNR
            self.y[i, :] += torch.normal(mean=0, std=torch.sqrt(var), size=(self.m,))

    def set_data(self, seed = 42):
        torch.manual_seed(seed)
        for i in range(self.size):
            self.set_tuple(i)
    
    def set_loader(self):
        return Data.DataLoader(dataset=Data.TensorDataset(self.y, self.x),
                               batch_size=self.batch_size,
                               shuffle=True)
    
import torch

class SyntheticSignalsOnline:
    def __init__(self, A, n, m, p=0.1, SNR=None, discretized=False):
        self.n = n  
        self.m = m  
        self.p = p  
        self.SNR = SNR  
        self.discretized = discretized  

        if A is None:                            
            self.A = self.A_initialization()          
        else:
            assert A.shape == (m, n), "A must have shape (m, n)"
            self.A = A

    def A_initialization(self):
        A = torch.normal(0, torch.sqrt(torch.tensor(1/self.m)), size=(self.m, self.n))
        return A / torch.linalg.norm(A, dim=0)  # Normalize columns

    def emit_data(self, size):
        """Generate sparse signals x and their corresponding linear projections y."""
        
        # Generate sparse indices
        mask = torch.rand(size, self.n) < self.p  # Boolean mask for sparsity

        # Generate sparse values
        if self.discretized:
            x = torch.where(mask, torch.randint(0, 2, (size, self.n), dtype=torch.float32) * 2 - 1, torch.tensor(0.0))
        else:
            x = torch.zeros(size, self.n)
            num_nonzero = mask.sum().item()
            x[mask] = torch.normal(0, 1, size=(num_nonzero,))  # Fill only non-zero indices

        # Generate linear projections
        y = x @ self.A.T  # Matrix multiplication, batch-wise

        # Add noise if SNR is specified
        if self.SNR is not None:
            var = torch.mean(y**2) / self.SNR  # Compute noise variance
            noise = torch.normal(0, torch.sqrt(var), size=y.shape)  # Generate noise
            y += noise  

        return y, x

        