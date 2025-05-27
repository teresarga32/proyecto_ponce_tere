"""This module contains function for input/output functionalities"""

import pandas as pd
from pathlib import Path
import numpy as np
import yfinance as yf
import json

def load_dataframe_from_csv_file(
    filename: Path = Path("training_log.csv"),
) -> pd.DataFrame:
    """Loads a pandas dataframe from a given Path
    
    Args:
        filename (str): A string containing the Path to the csv file containing data to be loaded.
        
    Returns:
        pd.Dataframe: A pandas dataframe containing the loaded data.
    """
    # Load your CSV log (update the filename if needed)
    log_df = pd.read_csv(filename)
    return log_df


def save_dataset_split(X_train, y_train, X_test, y_test, filename="multiasset_data.npz"):
    np.savez_compressed(filename, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Saved dataset to {filename}")


def load_dataset_split(filename="multiasset_data.npz"):
    data = np.load(filename)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    print(f"Loaded dataset from {filename}")
    return X_train, y_train, X_test, y_test


def download_and_extract_returns_from_tickers(
    tickers: list[str],
    period: str = "20y"
) -> pd.DataFrame:
    all_returns = []
    for ticker in tickers:
        print(f"Downloading {ticker} data...")
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            continue
        returns = data['Close'].pct_change().dropna() * 100
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        returns.name = ticker
        all_returns.append(returns)
    returns_df = pd.concat(all_returns, axis=1).dropna()
    return returns_df


def save_dict_to_json(data: dict, filename: str) -> None:
    """
    Saves a dictionary to a JSON file.

    Parameters:
        data (dict): The dictionary to save.
        filename (str): The path to the JSON file.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Dictionary successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving dictionary to JSON: {e}")
