import torch

import warnings
warnings.filterwarnings("ignore")

from src.models.ALISTA import ALISTA
from src.models.LISTA_CPSS import LISTA_CPSS

from src.models.ALDCISTA import ALDC_ISTA
from src.models.L_DC_ISTA_CPSS import L_DC_ISTA_CPSS

from src.models.LePOM import LePOM_CP
from src.models.ALePOM import ALePOM

from src.utils.train import layerwise_train
from src.utils.synthetic_data import SyntheticSignalsOnline

n_ = 500
m_ = 250
p_ = 0.2

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

T = 32
lambd_0 = 0.4

A_lr = 1e-3
A_ft_lr = 1e-3

DD_lr = 5e-4
DD_ft_lr = 5e-4

# Training the models

print('------------Starting training ALISTA------------')
model_1 = ALISTA(torch.clone(A_), lambd_0, T = T, SS = True)
layerwise_train(model_1, 'C', 'ALISTA', 'Online', generator, lr = A_lr, ft_lr = A_ft_lr, verbose=True)
torch.save(model_1.state_dict(), r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_1_RECONSTRUCTION_weights.pth")

print('------------Starting training LISTA-CPSS------------')
model_2 = LISTA_CPSS(torch.clone(A_), lambd_0, T = T, SS = True)
layerwise_train(model_2, 'C', 'LISTA-CPSS', 'Online', generator, lr = DD_lr, ft_lr =  DD_ft_lr, verbose=True)
torch.save(model_2.state_dict(), r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_2_RECONSTRUCTION_weights.pth")

print('------------Starting training ALDCISTA------------')
model_3 = ALDC_ISTA(torch.clone(A_), 'MCP', lambd_0, T = T, W = torch.clone(model_1.W).to('cuda:0'), SS = True)
layerwise_train(model_3, 'DC', 'AL-DC-ISTA', 'Online', generator, 'MCP', lr = A_lr, ft_lr = A_ft_lr, verbose=True)
torch.save(model_3.state_dict(), r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_3_RECONSTRUCTION_weights.pth")

print('------------Starting training LDCISTACPSS------------')
model_4 = L_DC_ISTA_CPSS(torch.clone(A_), 'MCP', lambd_0, T = T, SS = True)
layerwise_train(model_4, 'DC', 'L-DC-ISTA-CPSS','Online', generator, 'MCP', lr = DD_lr, ft_lr =  DD_ft_lr, verbose=True)
torch.save(model_4.state_dict(), r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_4_RECONSTRUCTION_weights.pth")

print('------------Starting training ALePOM------------')
model_5 = ALePOM(torch.clone(A_), lambd_0, T = T, W = torch.clone(model_1.W).to('cuda:0'))
layerwise_train(model_5, 'POM', 'ALePOM', 'Online', generator, lr = A_lr, ft_lr = A_ft_lr, verbose=True)
torch.save(model_5.state_dict(), r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_5_RECONSTRUCTION_weights.pth")

print('------------Starting training LePOM------------')
model_6 = LePOM_CP(torch.clone(A_), lambd_0, T = T)
layerwise_train(model_6, 'POM', 'LePOM', 'Online', generator, lr = DD_lr, ft_lr = DD_ft_lr, verbose=True)
torch.save(model_6.state_dict(), r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/model_6_RECONSTRUCTION_weights.pth")

torch.save(A_, r"C:/Users/Leonardo/Documents/GitHub/ModelBasedDL4SCA/ckpts/synthetic_sensing.pth")
