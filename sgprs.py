import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict
from .utils import normalize_series

def pagerank_importance(G: nx.Graph) -> pd.Series:
    pr = nx.pagerank(G, alpha=0.85)
    pr = pd.Series(pr, name="pagerank")
    imp = pr / pr.max() if pr.max() else pr * 0.0
    imp.name = "R_imp"
    return imp

def simrank(G: nx.Graph, C: float = 0.8, max_iter: int = 10, eps: float = 1e-4) -> Dict:
    """
    Simple SimRank implementation. Returns dict of dicts s[u][v] ~ similarity.
    """
    nodes = list(G.nodes())
    idx = {n:i for i,n in enumerate(nodes)}
    n = len(nodes)
    S = np.eye(n)
    A = nx.to_numpy_array(G, nodelist=nodes)
    # Normalize by degree for neighbors
    deg = A.sum(axis=1)
    for it in range(max_iter):
        S_new = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                Ni = np.where(A[i] > 0)[0]
                Nj = np.where(A[j] > 0)[0]
                if len(Ni)==0 or len(Nj)==0:
                    sij = 0.0
                else:
                    sij = C * np.sum(S[np.ix_(Ni, Nj)]) / (len(Ni) * len(Nj))
                S_new[i, j] = sij
                S_new[j, i] = sij
        if np.linalg.norm(S_new - S, ord="fro") < eps:
            S = S_new
            break
        S = S_new
    # Convert to dict
    out = {u:{} for u in nodes}
    for i,u in enumerate(nodes):
        for j,v in enumerate(nodes):
            out[u][v] = float(S[i,j])
    return out

def structural_risk(G: nx.Graph, sim: Dict, neighbor_risk: pd.Series) -> pd.Series:
    """
    R_struct(u) = avg_{v in N(u)} sim(u,v) * R_neighbor(v)
    """
    svals = {}
    for u in G.nodes():
        Nu = list(G.neighbors(u))
        if not Nu:
            svals[u] = 0.0
            continue
        acc = 0.0
        for v in Nu:
            acc += sim[u].get(v, 0.0) * float(neighbor_risk.get(v, 0.0))
        svals[u] = acc / len(Nu)
    s = pd.Series(svals, name="R_struct")
    return normalize_series(s)

def compute_sgprs(
    G: nx.Graph,
    aprs: pd.Series,
    w_struct: float = 0.55,
    w_imp: float = 0.45,
    sim_C: float = 0.8,
    sim_iter: int = 10
) -> pd.Series:
    """
    SGPRS(u) = w_struct * R_struct(u) + w_imp * R_imp(u)
    - R_struct from SimRank-weighted neighbor APRS (normalized)
    - R_imp from normalized PageRank
    Default weights reflect higher correlation of structural similarity with APRS.
    """
    pr_imp = pagerank_importance(G)
    sim = simrank(G, C=sim_C, max_iter=sim_iter)
    r_struct = structural_risk(G, sim, aprs)
    # align indices
    df = pd.concat([r_struct, pr_imp], axis=1).fillna(0.0)
    sgprs = w_struct * df["R_struct"] + w_imp * df["R_imp"]
    sgprs.name = "SGPRS"
    return sgprs
