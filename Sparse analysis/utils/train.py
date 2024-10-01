import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim

#####################################
######## END-TO-END TRAINING ########
#####################################


def train(model, train_loader, valid_loader, lr=5e-3, num_epochs=100, verbose=True, clip_value=10.0, eps=1e-6):
    # Automatically set device to 'cuda' if available, otherwise 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Move the model to the appropriate device

    # Use Adam optimizer with custom epsilon for stability
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        eps=eps,  # Adjust the tolerance for Adam
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.1
    )

    loss_train = np.zeros((num_epochs,))
    loss_test = np.zeros((num_epochs,))

    # Main train loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        signal_power_train = 0  # Track the signal power

        for _, (Y, S) in enumerate(train_loader):
            # Move data to the appropriate device (CPU or GPU)
            Y = Y.to(device)
            S = S.to(device)

            S_hat = model.forward(Y)
            
            # Calculate MSE and signal power
            mse_loss = F.mse_loss(S_hat, S, reduction="sum")
            signal_power = torch.sum(S ** 2)
            
            # Accumulate squared error and signal power
            train_loss += mse_loss.item()
            signal_power_train += signal_power.item()

            optimizer.zero_grad()
            mse_loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            
            model.zero_grad()
        
        # Compute NMSE in dB for training set
        nmse_train = train_loss / (signal_power_train + eps)
        loss_train[epoch] = 10 * np.log10(nmse_train + eps)
        scheduler.step()

        # Validation
        model.eval()
        test_loss = 0
        signal_power_test = 0  # Track signal power for validation

        with torch.no_grad():  # No gradient calculation needed during validation
            for _, (Y, S) in enumerate(valid_loader):
                # Move data to the appropriate device
                Y = Y.to(device)
                S = S.to(device)

                S_hat = model.forward(Y)
                
                # Calculate MSE and signal power
                mse_loss = F.mse_loss(S_hat, S, reduction="sum")
                signal_power = torch.sum(S ** 2)
                
                # Accumulate squared error and signal power for validation set
                test_loss += mse_loss.item()
                signal_power_test += signal_power.item()
        
        # Compute NMSE in dB for validation set
        nmse_test = test_loss / (signal_power_test + eps)
        loss_test[epoch] = 10 * np.log10(nmse_test + eps)

        # Log progress
        if epoch % 10 == 0 and verbose:
            print(
                "Epoch %d, Train NMSE (dB) %.8f, Validation NMSE (dB) %.8f"
                % (epoch, loss_train[epoch], loss_test[epoch])
            )

    return loss_test

#####################################
######## LAYER-WISE TRAINING ########
#####################################

def layerwise_train(
        model, 
        train_loader, 
        valid_loader, 
        model_class,
        lr=5e-4, 
        ft_lr = 3e-5,
        num_epochs=100, 
        verbose=True, 
        clip_value=10.0, 
        eps=1e-6, 
        fine_tune_epochs=100):

    assert model_class in [
        'LISTA',
        'LISTA-CPSS',
        'TILISTA',
        'ALISTA'
    ]

    device = model.device
    model = model.to(device)

    linear_shared = model.linear_shared

    loss_train_all = {}
    loss_test_all = {}

    T = model.T  # Number of layers/iterations

    for t in range(T):
        if verbose:
            print(f"===== Training Layer {t+1}/{T} =====")

        # Freeze parameters beyond current layer t
        for layer_idx in range(T + 1):
            if layer_idx > t:
                model.beta[layer_idx].requires_grad = False
                if model_class != 'LISTA' and linear_shared == True:
                    model.mu[layer_idx].requires_grad = False
                if not linear_shared:
                    if model_class == 'LISTA':
                        model.Ws_1[layer_idx].weight.requires_grad = False
                        model.Ws_2[layer_idx].weight.requires_grad = False
                    if model_class == 'LISTA-CPSS':
                        model.Ws[layer_idx].weight.requires_grad = False            
            else:
                model.beta[layer_idx].requires_grad = True
                if model_class != 'LISTA' and linear_shared == True:
                    model.mu[layer_idx].requires_grad = True
                if not linear_shared:
                    if model_class == 'LISTA':
                        model.Ws_1[layer_idx].weight.requires_grad = True
                        model.Ws_2[layer_idx].weight.requires_grad = True
                    if model_class == 'LISTA-CPSS':
                        model.Ws[layer_idx].weight.requires_grad = True

        # Define optimizer to include W1, W2, and parameters up to layer t
        if model_class == 'LISTA':
            if not linear_shared:
                optimizer = optim.Adam(
                    list(model.Ws_1.parameters())[:t+1] + 
                    list(model.Ws_2.parameters())[:t+1] + 
                    list(model.beta)[:t+1],
                    lr=lr,
                    eps=eps
                )
            else:
                optimizer = optim.Adam(
                    list(model.Ws_1.parameters()) + 
                    list(model.Ws_2.parameters()) + 
                    list(model.beta)[:t+1] + 
                    list(model.mu)[:t+1],
                    lr=lr,
                    eps=eps
                )

        if model_class == 'LISTA-CPSS':
            optimizer = optim.Adam(
                list(model.Ws.parameters())[:t+1] +
                list(model.beta)[:t+1] + 
                list(model.mu)[:t+1],
                lr=lr,
                eps=eps
            )

        if model_class == 'TILISTA':
            optimizer = optim.Adam(
                list(model.W.parameters())[:t+1] +
                list(model.beta)[:t+1] + 
                list(model.mu)[:t+1],
                lr=lr,
                eps=eps
            )

        if model_class == 'ALISTA':
            optimizer = optim.Adam(
                [model.beta[layer_idx] for layer_idx in range(t+1)] + 
                [model.mu[layer_idx] for layer_idx in range(t+1)],
                lr=lr,
                eps=eps
            )

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        # Initialize loss tracking for current layer
        loss_train = np.zeros((num_epochs,))
        loss_test = np.zeros((num_epochs,))

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            signal_power_train = 0.0

            for Y, S in train_loader:
                Y = Y.to(device)
                S = S.to(device)

                optimizer.zero_grad()
                S_hat = model(Y, its=t)

                mse_loss = F.mse_loss(S_hat, S, reduction="sum")
                signal_power = torch.sum(S ** 2)

                loss = mse_loss
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()

                train_loss += mse_loss.item()
                signal_power_train += signal_power.item()

            # Compute NMSE in dB for training set
            nmse_train = train_loss / (signal_power_train + eps)
            loss_train[epoch] = 10 * np.log10(nmse_train + eps)
            scheduler.step()

            # Validation
            model.eval()
            test_loss_epoch = 0.0
            signal_power_test = 0.0

            with torch.no_grad():
                for Y, S in valid_loader:
                    Y = Y.to(device)
                    S = S.to(device)

                    S_hat = model(Y, its=t)

                    mse_loss = F.mse_loss(S_hat, S, reduction="sum")
                    signal_power = torch.sum(S ** 2)

                    test_loss_epoch += mse_loss.item()
                    signal_power_test += signal_power.item()

            nmse_test = test_loss_epoch / (signal_power_test + eps)
            loss_test[epoch] = 10 * np.log10(nmse_test + eps)

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(
                    f"Layer {t+1}, Epoch {epoch+1}/{num_epochs}, "
                    f"Train NMSE (dB): {loss_train[epoch]:.6f}, "
                    f"Validation NMSE (dB): {loss_test[epoch]:.6f}"
                )

        # Store losses for current layer
        loss_train_all[f"Layer_{t+1}"] = loss_train
        loss_test_all[f"Layer_{t+1}"] = loss_test

        if verbose:
            print(f"===== Finished Training Layer {t+1}/{T} =====\n")

        # Optionally, fine-tune the entire network
        if fine_tune_epochs > 0:
            if verbose:
                print("===== Fine-Tuning the Entire Network =====")

            # Unfreeze all parameters
            for layer_idx in range(t + 1):
                model.beta[layer_idx].requires_grad = True
                if model_class != 'LISTA' and linear_shared == True:
                    model.mu[layer_idx].requires_grad = True
                if not linear_shared:
                    if model_class == 'LISTA':
                        model.Ws_1[layer_idx].weight.requires_grad = True
                        model.Ws_2[layer_idx].weight.requires_grad = True
                    if model_class == 'LISTA-CPSS':
                        model.Ws[layer_idx].weight.requires_grad = True   

            if model_class == 'LISTA':
                if not linear_shared:
                    optimizer = optim.Adam(
                        list(model.Ws_1.parameters())[:t+1] + 
                        list(model.Ws_2.parameters())[:t+1] + 
                        [model.beta[layer_idx] for layer_idx in range(t+1)],
                        lr=ft_lr,
                        eps=eps
                    )
                else:
                    optimizer = optim.Adam(
                        list(model.Ws_1.parameters()) + 
                        list(model.Ws_2.parameters()) + 
                        [model.beta[layer_idx] for layer_idx in range(t+1)] + 
                        [model.mu[layer_idx] for layer_idx in range(t+1)],
                        lr=ft_lr,
                        eps=eps
                    )

            if model_class == 'LISTA-CPSS':
                optimizer = optim.Adam(
                    list(model.Ws.parameters())[:t+1] +
                    [model.beta[layer_idx] for layer_idx in range(t+1)] + 
                    [model.mu[layer_idx] for layer_idx in range(t+1)],
                    lr=ft_lr,
                    eps=eps
                )

            if model_class == 'TILISTA':
                optimizer = optim.Adam(
                    list(model.W.parameters())[:t+1] +
                    [model.beta[layer_idx] for layer_idx in range(t+1)] + 
                    [model.mu[layer_idx] for layer_idx in range(t+1)],
                    lr=ft_lr,
                    eps=eps
                )

            if model_class == 'ALISTA':
                optimizer = optim.Adam(
                    [model.beta[layer_idx] for layer_idx in range(t+1)] + 
                    [model.mu[layer_idx] for layer_idx in range(t+1)],
                    lr=ft_lr,
                    eps=eps
                )

            scheduler_ft = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

            # Initialize loss tracking for fine-tuning
            loss_train_ft = np.zeros((fine_tune_epochs,))
            loss_test_ft = np.zeros((fine_tune_epochs,))

            for epoch in range(fine_tune_epochs):
                model.train()
                train_loss = 0.0
                signal_power_train = 0.0

                for Y, S in train_loader:
                    Y = Y.to(device)
                    S = S.to(device)

                    optimizer.zero_grad()
                    S_hat = model(Y, its=t)

                    mse_loss = F.mse_loss(S_hat, S, reduction="sum")
                    signal_power = torch.sum(S ** 2)

                    loss = mse_loss
                    loss.backward()

                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                    optimizer.step()

                    train_loss += mse_loss.item()
                    signal_power_train += signal_power.item()

                # Compute NMSE in dB for training set
                nmse_train = train_loss / (signal_power_train + eps)
                loss_train_ft[epoch] = 10 * np.log10(nmse_train + eps)
                scheduler_ft.step()

                # Validation
                model.eval()
                test_loss_epoch = 0.0
                signal_power_test = 0.0

                with torch.no_grad():
                    for Y, S in valid_loader:
                        Y = Y.to(device)
                        S = S.to(device)

                        S_hat = model(Y, its=t)

                        mse_loss = F.mse_loss(S_hat, S, reduction="sum")
                        signal_power = torch.sum(S ** 2)

                        test_loss_epoch += mse_loss.item()
                        signal_power_test += signal_power.item()

                nmse_test = test_loss_epoch / (signal_power_test + eps)
                loss_test_ft[epoch] = 10 * np.log10(nmse_test + eps)

                if verbose and (epoch % 10 == 0 or epoch == fine_tune_epochs - 1):
                    print(
                        f"Fine-Tune Epoch {epoch+1}/{fine_tune_epochs}, "
                        f"Train NMSE (dB): {loss_train_ft[epoch]:.6f}, "
                        f"Validation NMSE (dB): {loss_test_ft[epoch]:.6f}"
                    )

            # Store fine-tuning losses
            loss_train_all["Fine_Tune"] = loss_train_ft
            loss_test_all["Fine_Tune"] = loss_test_ft

            if verbose:
                print("===== Finished Fine-Tuning =====\n")

    return loss_train_all, loss_test_all
