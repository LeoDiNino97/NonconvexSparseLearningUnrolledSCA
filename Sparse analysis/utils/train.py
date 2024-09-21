import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F

def train(model, train_loader, valid_loader, lr=5e-3, num_epochs=100, verbose=True, clip_value=10.0, eps=1e-6):
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

        for _, (Y, S) in enumerate(valid_loader):
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
