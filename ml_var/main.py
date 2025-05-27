import torch
import pandas as pd

from ml_var.preprocessing import (
    download_and_preprocess_multiasset,
)
from ml_var.train import train_model
from ml_var.visualization import plot_metrics
from ml_var.io import (
    save_dataset_split, 
    load_dataset_split,
    save_dict_to_json,
)

from pathlib import Path

import argparse

def parse_arguments() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Specify which embeddings to search.")
    parser.add_argument(
        "--download_data",
        action="store_true",
        default=False,
        help="Whether to download the specified data or load it from an existing file.",
    )
    parser.add_argument(
        "--lookback_period",
        type=int,
        default=10,
        help="The desired lookback period to use.",
    )
    parser.add_argument(
        "--rolling_garch",
        action="store_false",
        default=True,
        help="Whether to use a rolling window for the garch estimation, defaults to False.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment_multi_asset_low_complexity_optimize_mdn_3_hs_32_lstm_1_mdn_8",
        help="The Path where the obtained training plots should be stored.",
    )
    return parser.parse_args()


def main():

    args = parse_arguments()

    # Preprocessing specifications
    lookback = args.lookback_period
    rolling_garch = args.rolling_garch
    download_data = args.download_data
    dataset_file = "" 
        
    experiment_name = args.experiment_name
    experiment_folder = Path(f"experiments/{experiment_name}")
    experiment_folder.mkdir(parents=True, exist_ok=True)

    if download_data:
        sp500_tickers = [
            # Tech & Growth
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", 
            #"COST", "ADBE", "CRM", "AMD", "INTC", "CSCO", "QCOM", "INTU",
            #"AMAT", "NOW", "TEAM", "SNPS", "CDNS", "KLAC", "ANET", "DASH",
            
            # Financials
            "JPM", "V", "MA", "BAC", "WFC", "C", "GS", "AXP", "BLK", "SPGI",
            #"PYPL", "COIN", "TROW", "NDAQ", "MSCI", "ICE", 
            
            # Healthcare
            "UNH", "LLY", "JNJ", "MRK", "ABBV", "PFE", "DHR", "MDT", "BMY",
            #"AMGN", "GILD", "VRTX", "REGN", "MRNA", "ILMN", "DXCM",
            
            # Industrials & Energy
            "RTX", "GE", "HON", "CAT", "DE", "LMT", "GD", "NOC", "BA", 
            #"UPS", "FDX", "CSX", "UNP", "XOM", "CVX", "COP", "SLB", "EOG",
            #"MPC", "PSX", "VLO", "EXE",
            
            # Consumer & Retail
            "WMT", "PG", "KO", "PEP", "COST", "TGT", "HD", "LOW", "NKE", 
            #"SBUX", "MCD", "YUM", "BKNG", "MAR", "WSM", "TKO",
            
            # Other Key Sectors
            "TSM", "ASML", "LIN", "APD", "ORCL", "SAP", "ADP", "IBM",
            #"TMO", "BDX", "ISRG", "ZTS", "IDXX", "BSX", "EW", "SYY",
            #"EL", "PM", "MO", "CL", "HSY", "KHC", "GIS"
        ]

        
        X_train, y_train, X_test, y_test, full_data = download_and_preprocess_multiasset(
            tickers=sp500_tickers,
            lookback=lookback,
            rolling_garch=rolling_garch,
        )
        
        # Ya tenemos nuestra data de training y testing
        
        save_dataset_split(X_train, y_train, X_test, y_test, filename="multiasset_data.npz")
        full_data.to_csv(experiment_folder / f"multi_asset_historical_data.csv")

    else:
        
        # Si tenemos ya data disponible, la podemos directamente loadear del archivo
        dataset_file = "multiasset_data.npz"
        X_train, y_train, X_test, y_test = load_dataset_split("multiasset_data.npz")


    # Aca deberiamos tener nuestra data de entrenamiento y testing
    # Para asi poder empezar el entrenamiento

    print(f"Training LSTM MDN model...")
    
    # Aca decidimos que especificaciones usar para entrenar nuestro modelo
    # Esta es la parte donde decidimos la arquitectura del modelo y los hyperparameters
    
    # Hyperparameters:
    training_specifications = {
        "epochs": 500, 
        "batch_size": 16, 
        "lr": 1e-4,
        "early_stopping_patience": 10,
    }
    
    # Arquitectura del modelo:
    architecture_specifications = {
        "input_size": 2,
        "num_lstm_layers": 1,
        "hidden_size": 32, 
        "mdn_size": 8,
        "n_components": 3,
        "dropout": 0.4,
        "bidirectional_lstm": False,
    }
    
    specifications = {
        "dataset_specifications": {
            "tickers_nr": 53,
            "dataset_file": dataset_file,
            "lookback": lookback,
            "rolling_garch": rolling_garch,
        },
        "training_specifications": training_specifications,
        "architecture_specifications": architecture_specifications,
    }
    
    save_dict_to_json(
        data=specifications, 
        filename=experiment_folder/f"{experiment_name}_specifications.json",
    )
    
    # Train model with given specifications
    model, scaler, log_df, best_epoch = train_model(
        X_train, 
        y_train, 
        X_test, 
        y_test,
        architecture_specifications=architecture_specifications,
        training_specifications=training_specifications,
        log_file=experiment_folder/f"{experiment_name}_training_log.csv",
    )

    # Store obtained model, scaler and logs
    torch.save(model.state_dict(), experiment_folder / f"{experiment_name}_multi_asset_var_model.pth")
    pd.to_pickle(scaler, experiment_folder / f"{experiment_name}_multi_asset_scaler.pkl")
    print(f"Training complete! Model and data saved for {experiment_name}")

    # Display computed metrics of trained model
    computed_metrics = [
        "test_loss",
        "test_mae",
        "test_mse",
        "exception_rate",
        "expected_shortfall",
        "kupiec_p",
        "christoffersen_p"
    ]
    best_epoch_metrics = log_df.loc[best_epoch-1].to_dict()
    
    print(f"\nDisplaying metrics of best epoch from training:\n")
    for metric in computed_metrics:
        metric_string = metric.replace('_', ' ').title()
        metric_score = best_epoch_metrics[metric]
        print(f"{metric_string}: {metric_score:.2f}")

    # Plot training metrics for visualization purposes
    plots_folder = experiment_folder / "training_plots"
    plots_folder.mkdir(parents=True, exist_ok=True)
    plot_metrics(
        log_df,
        plots_folder,
    )

if __name__ == "__main__":
    main()
