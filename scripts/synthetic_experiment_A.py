import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

import pickle

import torch
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from src.models.ALISTA import ALISTA
from src.models.LISTA_CPSS import LISTA_CPSS

from src.models.ALDCISTA import ALDC_ISTA
from src.models.L_DC_ISTA_CPSS import L_DC_ISTA_CPSS

from src.models.LePOM import LePOM_CP
from src.models.ALePOM import ALePOM

from src.utils.train import layerwise_train
from src.utils.synthetic_data import SyntheticSignals, SyntheticSignalsOnline

n_ = 500
m_ = 250
p_ = 0.1

torch.random.manual_seed(42)
A_ = torch.normal(0, torch.sqrt(torch.tensor(1/m_)), size = (m_, n_))
A_ /= torch.linalg.norm(A_, dim = 0)


generator = SyntheticSignalsOnline(
        A = A_,
        n = n_,        
        m = m_,
        p = p_,
        SNR = None
        )

T = 16
lambd_0 = 0.4

A_lr = 1e-3
A_ft_lr = 1e-3

DD_lr = 5e-4
DD_ft_lr = 5e-4

# Training the models

print('------------Starting training ALISTA------------')
model_1 = ALISTA(torch.clone(A_), lambd_0, T = T, SS = True)
layerwise_train(model_1, 'C', 'ALISTA', generator, lr = A_lr, ft_lr = A_ft_lr, verbose=True)

print('------------Starting training LISTA-CPSS------------')
model_2 = LISTA_CPSS(torch.clone(A_), lambd_0, T = T, SS = True)
layerwise_train(model_2, 'C', 'LISTA-CPSS', generator, lr = DD_lr, ft_lr =  DD_ft_lr, verbose=True)

print('------------Starting training ALDCISTA------------')
model_3 = ALDC_ISTA(torch.clone(A_), 'PNEG', lambd_0, T = T, W = torch.clone(model_1.W).to('cuda:0'), SS = False)
layerwise_train(model_3, 'DC', 'AL-DC-ISTA', generator, 'PNEG', lr = A_lr, ft_lr = A_ft_lr, verbose=True)

print('------------Starting training LDCISTACPSS------------')
model_4 = L_DC_ISTA_CPSS(torch.clone(A_), 'PNEG', lambd_0, T = T, SS = True)
layerwise_train(model_4, 'DC', 'L-DC-ISTA-CPSS', generator, 'PNEG', lr = DD_lr, ft_lr =  DD_ft_lr, verbose=True)

print('------------Starting training ALePOM------------')
model_5 = ALePOM(torch.clone(A_), lambd_0, T = T, W = torch.clone(model_1.W).to('cuda:0'))
layerwise_train(model_5, 'POM', 'ALePOM', generator, lr = A_lr, ft_lr = A_ft_lr, verbose=True)

print('------------Starting training LePOM------------')
model_6 = LePOM_CP(torch.clone(A_), lambd_0, T = T)
layerwise_train(model_6, 'POM', 'LePOM', generator, lr = DD_lr, ft_lr = DD_ft_lr, verbose=True)


print('------------Saving models checkpoints------------')
# Store models checkpoints
torch.save(model_1.state_dict(), r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_1_NOISY_weights.pth")
torch.save(model_2.state_dict(), r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_2_NOISY_weights.pth")
torch.save(model_3.state_dict(), r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_3_NOISY_weights.pth")
torch.save(model_4.state_dict(), r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_4_NOISY_weights.pth")
torch.save(model_5.state_dict(), r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_5_NOISY_weights.pth")
torch.save(model_6.state_dict(), r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_6_NOISY_weights.pth")

print('------Testing models in noisy scenarios--------')
# Test them in noisy scenarios
SNRs = [1] + list(range(5,100,5))

noisy_scenarios = {SNR: {
    'ALISTA':0,
    'LISTA-CPSS':0,
    'AL-DC-ISTA-PNEG':0,
    'L-DC-ISTA-CPSS-PNEG':0,
    'ALePOM':0,
    'LePOM':0
} for SNR in SNRs}


for SNR_ in tqdm(SNRs):
    for _ in range(10):
        test_set = SyntheticSignals(
            A = A_,
            n = n_,        
            m = m_,
            p = p_,
            SNR = 10**(SNR_/10),
            size = 1000
            ).set_loader()

        noisy_scenarios[SNR_]['ALISTA'] += model_1.compute_nmse_inference(test_set)[-1].cpu()
        noisy_scenarios[SNR_]['LISTA-CPSS'] += model_2.compute_nmse_inference(test_set)[-1].cpu()

        noisy_scenarios[SNR_]['AL-DC-ISTA-PNEG'] += model_3.compute_nmse_inference(test_set)[-1].cpu()
        noisy_scenarios[SNR_]['L-DC-ISTA-CPSS-PNEG'] += model_4.compute_nmse_inference(test_set)[-1].cpu()

        noisy_scenarios[SNR_]['ALePOM'] += model_5.compute_nmse_inference(test_set)[-1].cpu()
        noisy_scenarios[SNR_]['LePOM'] += model_6.compute_nmse_inference(test_set)[-1].cpu()

# Store results
with open(r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/results/reconstruction_noisy.pkl", 'wb') as handle:
    pickle.dump(noisy_scenarios, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Visualization
# Set global font
plt.rcParams["font.family"] = "Times New Roman"

# Increase figure size for better visibility
plt.figure(figsize=(10, 6), dpi=600)

models = [
    'ALISTA',
    'LISTA-CPSS',
    'AL-DC-ISTA-PNEG',
    'L-DC-ISTA-CPSS-PNEG',
    'ALePOM',
    'LePOM'
]

linestyles = [
    "-",    # Solid
    "-",   # Dashed
    "--",    # Solid
    "--",   # Dashed
    ":",
    ":"]

markers = ["o", "s", "o", "s", "o", "s"]
colors = [
    [0, 0.4470, 0.7410],   
    [0, 0.4470, 0.7410],  
    [0.8500, 0.3250, 0.0980],  
    [0.8500, 0.3250, 0.0980],  
    [0.9290, 0.6940, 0.1250],  
    [0.9290, 0.6940, 0.1250]   
]


# Plot each model
for i, model_ in enumerate(models):
    plt.plot(SNRs, 
             [noisy_scenarios[SNR_][model_] / 10 for SNR_ in SNRs], 
             label=models[i], 
             color=colors[i], 
             marker=markers[i], 
             linestyle=linestyles[i],
             markersize=6, linewidth=2.5)
plt.xticks(SNRs)

# Axis labels
plt.ylabel('NMSE (dB)', fontsize=14)
plt.xlabel('SNR (db)', fontsize=14)

# Beautify the grid
plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)

# Adjust legend placement
plt.legend(fontsize=12, fancybox=True, shadow=True, frameon=True, loc='upper right')

# Save the plot
plt.savefig(r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/plots/reconstruction_error_1.png", dpi=600, bbox_inches="tight", format="png")  

# Show the plot
plt.show()

print('--------Testing models in noiseless scenarios-------')
noiseless_scenarios = {
    'ALISTA':np.zeros(T),
    'LISTA-CPSS':np.zeros(T),
    'AL-DC-ISTA-PNEG':np.zeros(T),
    'L-DC-ISTA-CPSS-PNEG':np.zeros(T),
    'ALePOM':np.zeros(T),
    'LePOM':np.zeros(T)
}


for _ in range(10):
    test_set = SyntheticSignals(
        A = A_,
        n = n_,        
        m = m_,
        p = p_,
        SNR = 10**(SNR_/10),
        size = 1000
        ).set_loader()

    noiseless_scenarios['ALISTA'] += model_1.compute_nmse_inference(test_set).cpu().numpy()
    noiseless_scenarios['LISTA-CPSS'] += model_2.compute_nmse_inference(test_set).cpu().numpy()

    noiseless_scenarios['AL-DC-ISTA-PNEG'] += model_3.compute_nmse_inference(test_set).cpu().numpy()
    noiseless_scenarios['L-DC-ISTA-CPSS-PNEG'] += model_4.compute_nmse_inference(test_set).cpu().numpy()

    noiseless_scenarios['ALePOM'] += model_5.compute_nmse_inference(test_set)[-1].cpu().numpy()
    noiseless_scenarios['LePOM'] += model_6.compute_nmse_inference(test_set)[-1].cpu().numpy()

# Store results
with open(r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/results/reconstruction_noiseless.pkl", 'wb') as handle:
    pickle.dump(noisy_scenarios, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Visualization
# Set global font
plt.rcParams["font.family"] = "Times New Roman"

# Increase figure size for better visibility
plt.figure(figsize=(10, 6), dpi=600)

models = [
    'ALISTA',
    'LISTA-CPSS',
    'AL-DC-ISTA-PNEG',
    'L-DC-ISTA-CPSS-PNEG',
    'ALePOM',
    'LePOM'
]

linestyles = [
    "-",    # Solid
    "-",   # Dashed
    "--",    # Solid
    "--",   # Dashed
    ":",
    ":"]

markers = ["o", "s", "o", "s", "o", "s"]
colors = [
    [0, 0.4470, 0.7410],   
    [0, 0.4470, 0.7410],  
    [0.8500, 0.3250, 0.0980],  
    [0.8500, 0.3250, 0.0980],  
    [0.9290, 0.6940, 0.1250],  
    [0.9290, 0.6940, 0.1250]   
]


# Plot each model
for i, model_ in enumerate(models):
    plt.plot(range(T), 
             noiseless_scenarios[model_] / 10 , 
             label=models[i], 
             color=colors[i], 
             marker=markers[i], 
             linestyle=linestyles[i],
             markersize=6, linewidth=2.5)
plt.xticks(range(T))

# Axis labels
plt.ylabel('NMSE (dB)', fontsize=14)
plt.xlabel('Layer index', fontsize=14)

# Beautify the grid
plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)

# Adjust legend placement
plt.legend(fontsize=12, fancybox=True, shadow=True, frameon=True, loc='upper right')

# Save the plot
plt.savefig(r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/plots/reconstruction_error_2.png", dpi=600, bbox_inches="tight", format="png")  

# Show the plot
plt.show()