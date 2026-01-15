# io_utils.py
import numpy as np

def zscore(x, ddof: int = 1):
    """
    Z-score a 1D array (or array-like) robustly.

    - Uses nanmean/nanstd so NaNs are ignored (important because peak windows can be padded with NaN).
    - Returns array with same shape as input.
    """
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=ddof)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (x - mu) / sd


def zscore_axis(X, axis=0, ddof: int = 1):
    """
    Z-score along a given axis, ignoring NaNs.
    Useful if you later want to reuse the similarity code.
    """
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=axis, keepdims=True)
    sd = np.nanstd(X, axis=axis, ddof=ddof, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (X - mu) / sd