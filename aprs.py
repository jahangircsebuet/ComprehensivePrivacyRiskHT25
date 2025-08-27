import numpy as np
import pandas as pd
from typing import Dict, List
from .data_io import ATTRIBUTES
from .utils import VIS_MAP, safe_div

def _compute_attribute_frequency(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    freqs = {}
    for a in ATTRIBUTES:
        freqs[a] = df[a].value_counts(dropna=False).to_dict()
    return freqs

def _afiuf(freq: int, m: int) -> float:
    # S(a) = f(a) / log2( m / f(a) + 1 )
    if freq <= 0:
        return 0.0
    return freq / (np.log2(m / freq + 1))

def _attribute_sensitivity_map(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Return per-attribute, per-value sensitivity using AFIUF."""
    m = len(df)
    freq_map = _compute_attribute_frequency(df)
    sens_map = {}
    for a, fmap in freq_map.items():
        sens_map[a] = {val: _afiuf(cnt, m) for val, cnt in fmap.items()}
    return sens_map

def _visibility_value(vis: str, friends: int, amax: int) -> float:
    v = VIS_MAP.get(vis, 1.0)
    if v is None:  # friends
        return safe_div(friends, amax, default=0.0)
    return v

def compute_aprs(
    attributes_df: pd.DataFrame,
    friends_count: Dict[str, int],
    amax: int
) -> pd.Series:
    """
    APRS(u) = sum_i S(a_i) * V(a_i(u))
    S(a_i) from AFIUF over value frequencies.
    V maps visibility: public=1, friends=friends_count/Amax, private=0.1.
    """
    sens_map = _attribute_sensitivity_map(attributes_df)
    scores = {}
    for _, row in attributes_df.iterrows():
        uid = row["user_id"]
        fcnt = friends_count.get(str(uid), 0)
        s = 0.0
        for a in ATTRIBUTES:
            val = row[a]
            vis = row.get(f"{a}_vis", "public")
            sval = sens_map[a].get(val, 0.0)
            vval = _visibility_value(vis, fcnt, amax)
            s += sval * vval
        scores[str(uid)] = s
    return pd.Series(scores, name="APRS")
