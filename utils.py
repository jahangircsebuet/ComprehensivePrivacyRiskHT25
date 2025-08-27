import numpy as np
import pandas as pd

VIS_MAP = {"public": 1.0, "friends": None, "private": 0.1}  # friends resolved later

def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if np.isclose(mx - mn, 0):
        return pd.Series(np.zeros_like(s), index=s.index)
    return (s - mn) / (mx - mn)

def safe_div(a, b, default=0.0):
    return a / b if b else default

def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default