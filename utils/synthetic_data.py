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
            self.var = torch.mean(self.y[i, :]**2) / self.SNR
            self.y[i, :] += torch.normal(mean=0, std=torch.sqrt(self.var), size=(self.m,))

    def set_data(self, seed = 42):
        torch.manual_seed(seed)
        for i in range(self.size):
            self.set_tuple(i)
    
    def set_loader(self):
        return Data.DataLoader(dataset=Data.TensorDataset(self.y, self.x),
                               batch_size=self.batch_size,
                               shuffle=True)
