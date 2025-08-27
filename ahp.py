import numpy as np
from typing import Tuple

def ahp_weights_content() -> Tuple[float, float, float]:
    # Matrix:
    # [ [1, 0.5, 0.33],
    #   [2, 1,   0.5 ],
    #   [3, 2,   1   ] ]
    A = np.array([[1, 0.5, 0.33],
                  [2, 1.0, 0.5 ],
                  [3, 2.0, 1.0 ]], dtype=float)
    colsum = A.sum(axis=0)
    norm = A / colsum
    w = norm.mean(axis=1)
    # Normalize to sum 1
    w = w / w.sum()
    return float(w[0]), float(w[1]), float(w[2])  # APRS, SGPRS, CBPRS

def ahp_weights_graph() -> Tuple[float, float, float]:
    # [ [1, 0.2, 0.33],
    #   [5, 1.0, 2.0 ],
    #   [3, 0.5, 1.0 ] ]
    A = np.array([[1, 0.2, 0.33],
                  [5, 1.0, 2.0 ],
                  [3, 0.5, 1.0 ]], dtype=float)
    colsum = A.sum(axis=0)
    norm = A / colsum
    w = norm.mean(axis=1)
    w = w / w.sum()
    return float(w[0]), float(w[1]), float(w[2])

def equal_weights() -> Tuple[float, float, float]:
    return 1/3, 1/3, 1/3
