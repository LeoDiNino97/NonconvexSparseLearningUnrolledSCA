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

from src.utils.synthetic_data import SyntheticSignals

# Models initialization 
A_ = torch.load(r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/synthetic_sensing.pth")
model_1 = ALISTA(torch.clone(A_), 0.1, T = 16, SS = True)
model_2 = LISTA_CPSS(torch.clone(A_), 0.1, T = 16, SS = True)
model_3 = ALDC_ISTA(torch.clone(A_), 'MCP', 0.1, T = 17, W = torch.clone(model_1.W).to('cuda:0'), SS = False)
model_4 = L_DC_ISTA_CPSS(torch.clone(A_), 'MCP', 0.1, T = 16, SS = False)
model_5 = ALePOM(torch.clone(A_), 0.1, T = 16, W = torch.clone(model_1.W).to('cuda:0'))
model_6 = LePOM_CP(torch.clone(A_), 0.1, T = 16)

model_1.load_state_dict(torch.load(r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_1_NOISY_weights.pth"))
model_2.load_state_dict(torch.load(r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_2_NOISY_weights.pth"))
model_3.load_state_dict(torch.load(r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_2b_NOISY_weights.pth"))
model_4.load_state_dict(torch.load(r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_4_NOISY_weights.pth"))
model_5.load_state_dict(torch.load(r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_5_NOISY_weights.pth"))
model_6.load_state_dict(torch.load(r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_6_NOISY_weights.pth"))

# Test them in noisy scenarios
SNRs = [1] + list(range(5,100,5))

noisy_scenarios = {SNR: {
    'ALISTA':0,
    'LISTA-CPSS':0,
    'AL-DC-ISTA':0,
    'L-DC-ISTA-CPSS':0,
    'ALePOM':0,
    'LePOM':0
} for SNR in SNRs}

n_ = 500
m_ = 250
p_ = 0.1
T = 16

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

        noisy_scenarios[SNR_]['AL-DC-ISTA'] += model_3.compute_nmse_inference(test_set)[-1].cpu()
        noisy_scenarios[SNR_]['L-DC-ISTA-CPSS'] += model_4.compute_nmse_inference(test_set)[-1].cpu()

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
    'AL-DC-ISTA',
    'L-DC-ISTA-CPSS',
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

print('--------Testing models in noiseless scenarios-------')
noiseless_scenarios = {
    'ALISTA':np.zeros(T),
    'LISTA-CPSS':np.zeros(T),
    'AL-DC-ISTA':np.zeros(T),
    'L-DC-ISTA-CPSS':np.zeros(T),
    'ALePOM':np.zeros(T),
    'LePOM':np.zeros(T)
}


for _ in range(10):
    test_set = SyntheticSignals(
        A = A_,
        n = n_,        
        m = m_,
        p = p_,
        SNR = None,
        size = 1000
        ).set_loader()

    noiseless_scenarios['ALISTA'] += model_1.compute_nmse_inference(test_set).cpu().numpy()
    noiseless_scenarios['LISTA-CPSS'] += model_2.compute_nmse_inference(test_set).cpu().numpy()

    noiseless_scenarios['AL-DC-ISTA'] += model_3.compute_nmse_inference(test_set).cpu().numpy()
    noiseless_scenarios['L-DC-ISTA-CPSS'] += model_4.compute_nmse_inference(test_set).cpu().numpy()

    noiseless_scenarios['ALePOM'] += model_5.compute_nmse_inference(test_set).cpu().numpy()
    noiseless_scenarios['LePOM'] += model_6.compute_nmse_inference(test_set).cpu().numpy()

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
    'AL-DC-ISTA',
    'L-DC-ISTA-CPSS',
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