"""Modulo incluyendo funciones para calcular volatilidades usando GARCH."""

from arch import arch_model
import numpy as np
import pandas as pd


def get_volatility_from_garch(
    returns: np.ndarray,
) -> np.ndarray:
    """Garch standard usando toda la data."""
    garch = arch_model(returns, vol='GARCH', p=1, q=1, dist='t')
    garch_fit = garch.fit(disp='off')
    volatility = garch_fit.conditional_volatility

    return volatility


def get_volatility_from_garch_rolling(
    returns: pd.Series,
    lookback: int,
) -> pd.Series:
    """
    Garch con rolling window usando solo periodos de lookback period.
    
    Compute rolling window GARCH(1,1) volatility forecasts.
    For each time t, fit GARCH on returns[t-lookback:t] and forecast volatility at t.
    Returns a pandas Series aligned with the input returns.
    """
    
    returns = pd.Series(returns.squeeze()).dropna()
    volatility = pd.Series(np.nan, index=returns.index)
    last_valid_vol = None

    for i in range(lookback, len(returns)):
        window_returns = returns.iloc[i-lookback:i]
        sigma = window_returns.std()
        if sigma < 1e-8:
            # Fallback: use last valid volatility or small constant
            volatility.iloc[i] = last_valid_vol if last_valid_vol is not None else 1e-3
            continue
        standardized_returns = (window_returns - window_returns.mean()) / sigma
        
        try:
            garch = arch_model(standardized_returns, vol='GARCH', p=1, q=1, dist='t', rescale=False)
            garch_fit = garch.fit(disp='off', options={'maxiter': 1000})
            forecast = garch_fit.forecast(horizon=1)
            vol = np.sqrt(forecast.variance.values[-1, 0]) * sigma
            if np.isnan(vol) or np.isinf(vol):
                raise ValueError("Volatility is NaN or Inf")
            volatility.iloc[i] = vol
            last_valid_vol = vol
            
        except Exception as e:
            # Fallback: EMA std or last valid value
            fallback_vol = window_returns.ewm(span=10).std().iloc[-1]
            if np.isnan(fallback_vol) or np.isinf(fallback_vol) or fallback_vol < 1e-8:
                fallback_vol = last_valid_vol if last_valid_vol is not None else 1e-3
            volatility.iloc[i] = fallback_vol
            last_valid_vol = fallback_vol

    return volatility
