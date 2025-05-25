"""This module includes functions to compute the backtesting metrics."""

import numpy as np

from scipy.stats import binomtest, chi2


def compute_christoffersen_score(
    exceptions: np.ndarray,
):
    # Christoffersen test (conditional coverage)
    # Count number of 00, 01, 10, 11 transitions
    exceptions = exceptions.astype(int)
    n00 = np.sum((exceptions[:-1] == 0) & (exceptions[1:] == 0))
    n01 = np.sum((exceptions[:-1] == 0) & (exceptions[1:] == 1))
    n10 = np.sum((exceptions[:-1] == 1) & (exceptions[1:] == 0))
    n11 = np.sum((exceptions[:-1] == 1) & (exceptions[1:] == 1))
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi1 = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0
    # Likelihood ratio test
    import math
    def safe_log(x): return math.log(x) if x > 0 else 0
    L0 = ((n01 + n11) * safe_log(pi1) + (n00 + n10) * safe_log(1 - pi1)) if pi1 not in [0,1] else 0
    L1 = (n01 * safe_log(pi01) + n00 * safe_log(1 - pi01) +
          n11 * safe_log(pi11) + n10 * safe_log(1 - pi11)) if all(p not in [0,1] for p in [pi01, pi11]) else 0
    LRcc = -2 * (L0 - L1)
    christoffersen_p = 1 - chi2.cdf(LRcc, 1)
    return christoffersen_p


def compute_exceptions(y_true, var_pred):
    exceptions = y_true < var_pred
    n = len(y_true)
    n_exceptions = exceptions.sum()
    exception_rate = n_exceptions / n

    return exceptions, n_exceptions, exception_rate


def compute_expected_shortfall(y_true, var_pred):
    exceptions = y_true < var_pred
    if exceptions.sum() == 0:
        return np.nan
    return y_true[exceptions].mean()


def var_backtesting(y_true, var_pred, alpha=0.01):
    """
    y_true: true returns (numpy array)
    var_pred: predicted VaR (numpy array)
    alpha: VaR level (e.g., 0.01 for 99% VaR)
    Returns: exception rate, Kupiec POF p-value, Christoffersen p-value
    """
    # Exception: when loss < VaR (for left tail)
    exceptions, n_exceptions, exception_rate = compute_exceptions(y_true, var_pred)
    # Kupiec POF test (unconditional coverage)
    # H0: exception_rate == alpha
    kupiec_p = binomtest(n_exceptions, len(y_true), alpha, alternative='two-sided').pvalue

    christoffersen_p = compute_christoffersen_score(
        exceptions=exceptions,
    )
    
    expected_shortfall = compute_expected_shortfall(
        y_true=y_true,
        var_pred=var_pred,
    )

    return {
        "exception_rate": float(exception_rate),
        "n_exceptions": float(n_exceptions),
        "kupiec_p": float(kupiec_p),
        "christoffersen_p": float(christoffersen_p),
        "expected_shortfall": float(expected_shortfall),
    }
