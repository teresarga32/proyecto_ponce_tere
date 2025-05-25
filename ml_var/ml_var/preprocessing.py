"""Modulo conteniendo funciones para preprocesar nuestra data para el entrenamiento."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bachelor_arbeit.io import download_and_extract_returns_from_tickers
from bachelor_arbeit.garch import (
    get_volatility_from_garch,
    get_volatility_from_garch_rolling,
)


def standardize_per_stock(returns_df):
    scalers = {}
    scaled_returns = pd.DataFrame(index=returns_df.index)
    for ticker in returns_df.columns:
        scaler = StandardScaler()
        # Reshape for sklearn, then flatten back to Series
        scaled = scaler.fit_transform(returns_df[ticker].values.reshape(-1, 1)).flatten()
        scaled_returns[ticker] = scaled
        scalers[ticker] = scaler
    return scaled_returns, scalers


def slice_data_into_lookback_periods(
    returns: np.ndarray,
    volatility: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide la data en slices que contienen solo el lookback period"""
    X, y = [], []
    for i in range(lookback, len(returns)):
        X.append(np.column_stack([
            returns.values[i-lookback:i], 
            volatility.values[i-lookback:i]
        ]))
        y.append(returns.values[i])

    X, y = np.array(X), np.array(y)
    X = X[lookback:]
    y = y[lookback:]

    return X, y


def prepare_multiasset_dataset(
    returns_df: pd.DataFrame,
    lookback: int,
    split_percentage: float = 0.8,
    rolling_garch: bool = True
):
    """Basado en los returns de cada ticker (stock):
        - Calcula sus volatilidades
        - Los separa en slices de lookback period
        - Los divide en training y testing datasets
        
        Returnea X (returns y volatilidades) e y (return del ultimo dia)
        para el training y testing
    """
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    for ticker in returns_df.columns:
        returns = returns_df[ticker]
        
        # Usamos GARCH para calcular las volatilidades de nuestra data
        if rolling_garch:
            volatility = get_volatility_from_garch_rolling(returns, lookback)
        else:
            volatility = get_volatility_from_garch(returns)
        # Aca ya tenemos nuestras volatilidades y returns de nuestra data
        
        # La data que tenemos la sliceamos en vectores de lookback periods
        X, y = slice_data_into_lookback_periods(returns, volatility, lookback)
        
        split = int(split_percentage * len(X))
        X_train_list.append(X[:split])
        y_train_list.append(y[:split])
        X_test_list.append(X[split:])
        y_test_list.append(y[split:])
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    return X_train, y_train, X_test, y_test


def download_and_preprocess_multiasset(
    tickers: list[str],
    lookback: int,
    rolling_garch: bool = True,
):

    # Descrgamos la data y extraemos los returns
    returns_df = download_and_extract_returns_from_tickers(tickers, period="20y")

    # Standardize per stock
    scaled_returns_df, scalers = standardize_per_stock(returns_df)
    
    X_train, y_train, X_test, y_test = prepare_multiasset_dataset(
        scaled_returns_df, lookback=lookback, rolling_garch=rolling_garch
    )

    return X_train, y_train, X_test, y_test, returns_df
