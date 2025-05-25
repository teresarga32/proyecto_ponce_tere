"""Modulo con funciones para el entranamiento de modelos."""
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

from ml_var.model import VaRModel

from ml_var.metrics import (
    mdn_loss,
    evaluate_model_with_backtesting,
)

import copy


def train_model(
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    architecture_specifications: dict[str, int | float],
    training_specifications: dict[str, int | float],
    log_file="training_log.csv"
):
    
    # Set training specifications:
    epochs = training_specifications["epochs"]
    batch_size = training_specifications["batch_size"]
    lr = training_specifications["lr"]
    early_stopping_patience = training_specifications["early_stopping_patience"]
    
    # Scalar data: centrar alrededor de 0 con standard deviation de 1:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, 2)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, 2)).reshape(X_test.shape)
    
    # Poner en un dataloader para el entrenamiento:
    train_data = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Crear modelo:
    model = VaRModel(
        input_size=architecture_specifications["input_size"],
        num_lstm_layers=architecture_specifications["num_lstm_layers"],
        hidden_size=architecture_specifications["hidden_size"], 
        mdn_size=architecture_specifications["mdn_size"], 
        n_components=architecture_specifications["n_components"], 
        dropout=architecture_specifications["dropout"],
        bidirectional_lstm=architecture_specifications["bidirectional_lstm"],
    )
    
    # Crear/definir Optimizer (e.g. Gradient Descent, Adam, etc.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # Crear Scheduler (esto hace que e.g. el learning rate se haga mas chico si la loss no baja en mucho tiempo)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # Comenzamos el entrenamiento con las especificaciones que definimos
    # Estas variables son para trackear nuestras metrics y nuestro mejor modelo
    logs = []
    best_loss = float('inf') # Siempre vamos a tomar el modelo con la menor loss para continuar el entrenamiento
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict()) # Y aca vamos a copiar siempre nuestro mejor modelo
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Entrenamos la cantidad definida de epochs que decidimos
        model.train()
        batch_losses = []
        # Vamos por toda la data en batches
        for batch_X, batch_y in loader:
            
            # Hacemos un forward pass
            pi, mu, sigma, _ = model(batch_X)
            
            # Calculamos la loss basada en los parametros de nuestro forward pass
            loss = mdn_loss(batch_y, pi, mu, sigma)
            optimizer.zero_grad()
            
            # Hacemos backward pass
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_loss = np.mean(batch_losses)
        epoch_std = np.std(batch_losses)


        # Cada 10 epochs evaluamos el modelo para ver como performea con el test dataset
        # Evaluate on test set at the end of each epoch (or every N epochs)
        do_eval = ((epoch+1) % 10 == 0 or epoch == epochs-1)
        if do_eval:
            
            # Calculamos las metrics para el test set
            test_loss, test_std, test_mae, test_mse, test_var, backtest = evaluate_model_with_backtesting(model, X_test, y_test, batch_size, alpha=0.01)

            # Guardamos las metrics en el log para analizar el proceso del entrenamiento despues
            logs.append({
                "epoch": epoch+1,
                "train_loss": epoch_loss,
                "train_loss_std": epoch_std,
                "test_loss": test_loss,
                "test_loss_std": test_std,
                "test_mae": test_mae,
                "test_mse": test_mse,
                **backtest,
            })
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f} | Test MSE: {test_mse:.4f} | Exception Rate: {backtest['exception_rate']:.4f} | Kupiec: {backtest['kupiec_p']:.4f} | Christoffersen: {backtest['christoffersen_p']:.4f}")
            scheduler.step(test_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")
            
            
            """Esto es early stopping:
            Implica que si el modelo no mejora en una cantidad de pasos (paciencia),
            Probablemente o ya hayamos llegado al minimo, o no vayamos a llegar,
            Entonces dejamos de entrenar ahi y nos quedamos con el mejor modelo hasta ese momento.
            """
            # Early stopping logic
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch+1
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch} with test loss {best_loss:.4f}")
                    break
            
        else:
            logs.append({
                "epoch": epoch+1,
                "train_loss": epoch_loss,
                "train_loss_std": epoch_std,
                "test_loss": np.nan,
                "test_loss_std": np.nan,
                "test_mae": np.nan,
                "test_mse": np.nan,
            })
        
    # Usamos siempre el mejor modelo - basandonos en la test loss
    # Restore best model
    model.load_state_dict(best_model_state)


    # Hacemos una ultima evaluacion del mejor modelo que obtuvimos
    # Final evaluation
    test_loss, test_std, test_mae, test_mse, test_var, backtest = evaluate_model_with_backtesting(model, X_test, y_test, batch_size, alpha=0.01)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f} | Test MSE: {test_mse:.4f}")
    print(f"VaR Exception Rate: {backtest['exception_rate']:.4f} | Kupiec p: {backtest['kupiec_p']:.4f} | Christoffersen p: {backtest['christoffersen_p']:.4f} | Expected Shortfall: {backtest['expected_shortfall']:.4f}")

    # Guardamos los logs para analizarlos y visualizarlos despues
    # Optionally, add to your log_df or save separately
    log_df = pd.DataFrame(logs)
    log_df.to_csv(log_file, index=False)
    print(f"Training log saved to {log_file}")
    
    # Returneamos el modelo, el scaler y los logs conteniendo las training metrics
    return model, scaler, log_df
