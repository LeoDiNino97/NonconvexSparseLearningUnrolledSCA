import wandb
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim

def get_params(model, 
               family, 
               model_class, 
               DCSIP, 
               phase, 
               t):
    
    params = []
    if phase == 'Layer':
        if family == 'DC':
            #params.append(model.lambd[t]) 
            params.append(model.theta[t])
            if model_class == 'AL-DC-ISTA':
                params.append(model.mu[t])
            if model_class == 'L-DC-ISTA-CPSS':
                params.append(list(model.Ws.parameters())[t])

            if DCSIP == 'PNEG':
                params.append(model.P[t])
            if DCSIP == 'SCAD':
                params.append(model.a[t])

        if family == 'C':
            params.append(model.beta[t])
            if model_class != 'LISTA' and model.linear_shared:
                params.append(model.mu[t])
            if not model.linear_shared:
                if model_class == 'LISTA':
                    params.append(list(model.Ws_1.parameters())[t])
                    params.append(list(model.Ws_2.parameters())[t])
                if model_class == 'LISTA-CPSS':
                    params.append(list(model.Ws.parameters())[t])

        if family == 'POM':
            params.append(model.beta[t])
            if model_class == 'ALePOM':
                params.append(model.mu[t])
            if model_class == 'LePOM':
                params.append(list(model.Ws.parameters())[t])
        
    if phase == 'Fine-tune':

        if family == 'DC':
            params += list(model.theta)[:t+1] 
            #params = list(model.lambd)[:t+1] + list(model.theta)[:t+1] 
            if model_class == 'AL-DC-ISTA':
                params += list(model.mu)[:t+1]
            if model_class == 'L-DC-ISTA-CPSS':
                params += list(model.Ws.parameters())[:t+1]
            if DCSIP == 'PNEG':
                params += list(model.P)[:t+1]
            if DCSIP == 'SCAD':
                params += list(model.a)[:t+1]

        if family == 'C':
            params = list(model.beta[:t+1])
            if model_class != 'LISTA' and model.linear_shared:
                params += list(model.mu[:t+1])
            if not model.linear_shared:
                if model_class == 'LISTA':
                    params += list(model.Ws_1.parameters())[:t+1]
                    params += list(model.Ws_2.parameters())[:t+1]
                if model_class == 'LISTA-CPSS':
                    params += list(model.Ws.parameters())[:t+1]   

        if family == 'POM':
            params = list(model.beta[:t+1])
            if model_class == 'ALePOM':
                params += list(model.mu[:t+1])
            if model_class == 'LePOM':
                params += list(model.Ws.parameters())[:t+1]

    # Disable gradient computation for all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradient computation only for the selected parameters
    for param in params:
        param.requires_grad = True

    return params

def train_ops(
        model, 
        generator, 
        Y_val,
        S_val,
        t, 
        train_batch_size, 
        optimizer, 
        scheduler,
        patience,
        max_epochs, 
        eps, 
        verbose):
    
    loss_train = []
    loss_test = []
    val_window = []
    
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    best_model_state = None

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        signal_power_train = 0.0

        Y, S = generator.emit_data(train_batch_size)
        Y = Y.to(model.device)
        S = S.to(model.device)

        optimizer.zero_grad()
        S_hat = model(y=Y, its=t, S = None)

        mse_loss = F.mse_loss(S_hat, S, reduction="sum")
        signal_power = torch.sum(S ** 2)

        loss = mse_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += mse_loss.item()
        signal_power_train += signal_power.item()

        nmse_train = train_loss / (signal_power_train + eps)
        loss_train.append(10 * np.log10(nmse_train + eps))
        
        # Log training loss to wandb
        wandb.log({"Train NMSE (dB)": loss_train[-1], "Epoch": epoch})
        
        if epoch % 10 == 0:
            model.eval()
            test_loss_epoch = 0.0
            signal_power_test = 0.0

            with torch.no_grad():
                S_hat = model(Y_val, its=t)
                mse_loss = F.mse_loss(S_hat, S_val, reduction="sum")
                signal_power = torch.sum(S_val ** 2)

                test_loss_epoch += mse_loss.item()
                signal_power_test += signal_power.item()

            nmse_test = test_loss_epoch / (signal_power_test + eps)
            loss_test.append(10 * np.log10(nmse_test + eps))
            val_window.append(loss_test[-1])
            
            # Log validation loss to wandb
            wandb.log({"Validation NMSE (dB)": loss_test[-1], "Epoch": epoch})

            if verbose and (epoch % 100 == 0):
                print(
                    f"Layer {t}, Step {epoch+1}, "
                    f"Train NMSE (dB): {loss_train[-1]:.6f}, "
                    f"Validation NMSE (dB): {loss_test[-1]:.6f}"
                )
            
            if epoch > patience:
                val_window.pop(0)                
                rel_improvement = abs((val_window[0] - val_window[-1]))
                if rel_improvement < 1e-3:
                    if verbose:
                        print(f"Early stopping triggered due to small improvement: {abs((val_window[0] - val_window[-1])):.6f}")
                    break

            if loss_test[-1] < best_val_loss:
                best_val_loss = loss_test[-1]
                epochs_since_improvement = 0
                best_model_state = model.state_dict()
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                model.load_state_dict(best_model_state)
                break
    
    return loss_train, loss_test, epoch

def layerwise_train(
        model, 
        family,
        model_class,
        generator,
        DCSIP = None,
        train_batch_size = 64,
        val_batch_size = 1000,
        lr=5e-4, 
        ft_lr=3e-5,
        max_epochs=200000, 
        verbose=True, 
        eps=1e-6, 
        ft_max_epochs=200000, 
        patience=2000):
    
    wandb.init(project="Deep Sparse Coder", config={
        "Scope": 'Sparse recovery',
        "Model class": model_class,
        "Family": family,
        "Learning Rate": lr,
        "Fine Tune Learning Rate": ft_lr,
        "Train Batch Size": train_batch_size,
        "Max Epoch": max_epochs,
        "Fine Tune Epochs": ft_max_epochs,
    })
    
    device = model.device
    model = model.to(device)
    loss_train_all = {}
    loss_test_all = {}
    T = model.T
    
    Y_val, S_val = generator.emit_data(val_batch_size)
    Y_val = Y_val.to(device)
    S_val = S_val.to(device)

    for t in range(T):
        if t > 0:
            lr *= 0.9
            ft_lr *= 0.9
            
        if verbose:
            print(f"===== Training Layer {t}/{T} =====")

        params = get_params(model, family, model_class, DCSIP, phase='Layer', t=t)
        optimizer = optim.Adam(params, lr=lr, eps=eps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

        loss_train, loss_test, epoch = train_ops(model, generator, Y_val, S_val,
                                          t, train_batch_size, optimizer, scheduler,
                                          patience, max_epochs, eps, verbose)

        loss_train_all[f"Layer_{t+1}"] = loss_train[:epoch+1]
        loss_test_all[f"Layer_{t+1}"] = loss_test[:epoch+1]
        
        if verbose:
            print(f"===== Finished Training Layer {t}/{T} =====\n")
        
        #wandb.log({"Layer": t+1, "Final Train NMSE (dB)": loss_train[-1], "Final Validation NMSE (dB)": loss_test[-1]})
        
        if ft_max_epochs > 0:
            if verbose:
                print("===== Fine-Tuning the Entire Network =====")
            params = get_params(model, family, model_class, DCSIP, phase='Fine-tune', t=t)
            optimizer_ft = optim.Adam(params, lr=ft_lr, eps=eps)
            scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=max_epochs, eta_min=1e-6)

            loss_train_ft, loss_test_ft, epoch = train_ops(model, generator, Y_val, S_val,
                                          t, train_batch_size, optimizer_ft, scheduler_ft,
                                          patience, ft_max_epochs, eps, verbose)
            
            loss_train_all["Fine_Tune"] = loss_train_ft[:epoch+1]
            loss_test_all["Fine_Tune"] = loss_test_ft[:epoch+1]
            
            #wandb.log({"Fine-tuning Epochs": epoch, "Final Fine-tune Train NMSE (dB)": loss_train_ft[-1], "Final Fine-tune Validation NMSE (dB)": loss_test_ft[-1]})
            
            if verbose:
                print("===== Finished Fine-Tuning =====\n")
    
    wandb.finish()
    return loss_train_all, loss_test_all



####################################################
######## LAYER-WISE TRAINING - UNSUPERVISED ########
####################################################

def train_U_ops(        
        model, 
        train_loader,
        valid_loader,
        t, 
        optimizer, 
        scheduler,
        patience,
        num_epochs, 
        eps, 
        verbose):
    # Initialize loss tracking for current layer
    loss_train = np.zeros((num_epochs,))
    loss_test = np.zeros((num_epochs,))

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        signal_power_train = 0.0

        for (Y, Y_noisy) in train_loader:
            Y = Y.to(model.device)
            Y_noisy = Y_noisy.to(model.device)

            optimizer.zero_grad()
            S_hat = model(y=Y_noisy, its=t, S = None)

            mse_loss = F.mse_loss(Y.T, torch.matmul(model.A, S_hat.T), reduction="sum")
            signal_power = torch.sum(Y ** 2)

            loss = mse_loss + 0.2 * torch.sum(torch.abs(S_hat))
            loss.backward()

            # Apply gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            scheduler.step()
            train_loss += mse_loss.item()
            signal_power_train += signal_power.item()

        # Compute NMSE in dB for training set
        nmse_train = train_loss / (signal_power_train + eps)
        loss_train[epoch] = 10 * np.log10(nmse_train + eps)
        wandb.log({"Train NMSE (dB)": loss_train[-1], "Epoch": epoch})

        # Validation
        model.eval()
        test_loss_epoch = 0.0
        signal_power_test = 0.0

        with torch.no_grad():
            for (Y, Y_noisy) in valid_loader:
                Y = Y.to(model.device)
                Y_noisy = Y_noisy.to(model.device)

                S_hat = model(y=Y_noisy, its=t, S = None)

                mse_loss = F.mse_loss(Y.T, torch.matmul(model.A, S_hat.T), reduction="sum")
                signal_power = torch.sum(Y ** 2)
                
                # Total loss with L1 penalty
                total_loss = mse_loss 

                test_loss_epoch += total_loss.item()
                signal_power_test += signal_power.item()

        nmse_test = test_loss_epoch / (signal_power_test + eps)
        loss_test[epoch] = 10 * np.log10(nmse_test + eps)
        scheduler.step(loss_test[epoch])
        wandb.log({"Validation NMSE (dB)": loss_test[-1], "Epoch": epoch})

        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(
                f"Layer {t+1}, Epoch {epoch+1}/{num_epochs}, "
                f"Train NMSE (dB): {loss_train[epoch]:.6f}, "
                f"Validation NMSE (dB): {loss_test[epoch]:.6f}"
            )

        # Early stopping check
        if loss_test[epoch] < best_val_loss:
            best_val_loss = loss_test[epoch]
            epochs_since_improvement = 0
            best_model_state = model.state_dict()  # Save the best model state
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            if verbose:
                print(f"Early stopping triggered after {epoch+1} epochs")
            model.load_state_dict(best_model_state)  # Restore best model state
            break
    return loss_train, loss_test, epoch

def layerwise_train_U(
        model, 
        family,
        train_loader, 
        valid_loader, 
        model_class,
        DCSIP = None,
        lr=5e-4, 
        ft_lr=3e-5,
        num_epochs=100, 
        verbose=True, 
        eps=1e-6, 
        fine_tune_epochs=100, 
        patience=30):

    wandb.init(project="Deep Sparse Coder", config={
        "Scope": 'BSD500 Denoising',
        "Model class": model_class,
        "Family": family,
        "Learning Rate": lr,
        "Fine Tune Learning Rate": ft_lr,
        "Max Epoch": num_epochs,
        "Fine Tune Epochs": num_epochs,
    })

    device = model.device
    model = model.to(device)

    loss_train_all = {}
    loss_test_all = {}
    T = model.T

    for t in range(T):

        if verbose:
            print(f"===== Training Layer {t+1}/{T} =====")
        params = get_params(model, family, model_class, DCSIP, phase='Layer', t=t)

        optimizer = optim.Adam(params, lr=lr, eps=eps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        loss_train, loss_test, epoch = train_U_ops(model, train_loader, valid_loader, t,
                                                   optimizer, scheduler, patience, 
                                                   num_epochs, eps, verbose)
        # Store losses for current layer
        loss_train_all[f"Layer_{t+1}"] = loss_train[:epoch+1]
        loss_test_all[f"Layer_{t+1}"] = loss_test[:epoch+1]

        if verbose:
            print(f"===== Finished Training Layer {t+1}/{T} =====\n")

        # Optionally, fine-tune the entire network
        if fine_tune_epochs > 0:
            if verbose:
                print("===== Fine-Tuning the Entire Network =====")

            params = get_params(model, family, model_class, DCSIP, phase='Fine-tune', t=t)

            #optimizer_ft = optim.AdamW(model.parameters(), lr=ft_lr, eps=eps, weight_decay=1e-2)
            optimizer_ft = optim.Adam(params, lr=ft_lr, eps=eps)
            scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=num_epochs, eta_min=1e-6)
            loss_train_ft, loss_test_ft, epoch = train_U_ops(model, train_loader, valid_loader, t,
                                                   optimizer_ft, scheduler_ft, patience, 
                                                   num_epochs, eps, verbose)

            # Store fine-tuning losses
            loss_train_all["Fine_Tune"] = loss_train_ft[:epoch+1]
            loss_test_all["Fine_Tune"] = loss_test_ft[:epoch+1]

            if verbose:
                print("===== Finished Fine-Tuning =====\n")

    return loss_train_all, loss_test_all


