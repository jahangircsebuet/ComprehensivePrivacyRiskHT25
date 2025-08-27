import os
import re
import math
import random
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

from .data_io import ATTRIBUTES
from .utils import VIS_MAP, safe_div

# ==== Helpers ===============================================================

SENSITIVE_ENTS = {"PERSON", "GPE", "LOC", "ORG", "DATE", "TIME", "MONEY", "EMAIL"}

def _load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

def _extract_entities(nlp, text: str) -> List[str]:
    ents = []
    if not nlp or not text:
        return ents
    doc = nlp(text)
    ents = [e.label_ for e in doc.ents]
    if "@" in text:
        at_hits = text.count("@")
        ents.extend(["EMAIL"] * at_hits)
    return ents

def _visibility_value(vis: str, friends: int, amax: int) -> float:
    v = VIS_MAP.get(vis, 1.0)
    if v is None:
        return safe_div(friends, amax, default=0.0)
    return v

# ==== 1) Attribute Inference Attack ========================================

def _neighbor_mode(values: List[str]) -> str:
    if not values:
        return None
    vc = pd.Series(values).value_counts()
    return vc.index[0]

def attribute_inference_accuracy(
    G: nx.Graph,
    attrs_df: pd.DataFrame,
    hidden_attr: str = "relationship",
    hide_rate: float = 0.5,
    seed: int = 13
) -> float:
    """
    Predict hidden_attr for users by looking at neighbors' values (simple neighbor-mode classifier).
    - Randomly hide 'hidden_attr' for 'hide_rate' of nodes (labels are still used for eval)
    - For each hidden node, predict as mode(neighbor values)
    Returns accuracy on hidden set.
    """
    rng = np.random.RandomState(seed)
    df = attrs_df.set_index("user_id")
    all_users = [u for u in G.nodes() if u in df.index]

    # choose hidden set
    hidden = set(rng.choice(all_users, size=int(len(all_users) * hide_rate), replace=False))
    y_true, y_pred = [], []

    for u in all_users:
        if u not in hidden:
            continue
        # neighbors' labels
        neigh = [n for n in G.neighbors(u) if n in df.index and n not in hidden]
        neigh_vals = [str(df.loc[n, hidden_attr]) for n in neigh if pd.notna(df.loc[n, hidden_attr])]
        guess = _neighbor_mode(neigh_vals)
        if guess is None:
            # fallback: global most frequent
            guess = df[hidden_attr].mode().iloc[0]
        y_true.append(str(df.loc[u, hidden_attr]))
        y_pred.append(guess)

    return float(accuracy_score(y_true, y_pred)) if y_true else 0.0

# ==== 2) Link Prediction Attack ============================================

def _score_pairs(G: nx.Graph, metric: str = "jaccard") -> List[Tuple[str, str, float]]:
    if metric == "jaccard":
        gen = nx.jaccard_coefficient(G)
    elif metric == "adamic_adar":
        gen = nx.adamic_adar_index(G)
    else:
        raise ValueError("metric must be 'jaccard' or 'adamic_adar'")
    return [(u, v, float(p)) for u, v, p in gen]

def link_prediction_recall(
    G: nx.Graph,
    sample_non_edges: int = 10000,
    k_ratio: float = 0.01,
    metric: str = "jaccard",
    seed: int = 7
) -> float:
    """
    Approx link-pred recall@K on hidden edges:
    - Randomly hide a small set of existing edges (E_hidden).
    - Score candidate pairs on the remaining graph.
    - Take top-K pairs and compute recall = |topK ∩ E_hidden| / |E_hidden|.
    """
    rng = np.random.RandomState(seed)
    if G.number_of_edges() == 0:
        return 0.0

    # copy graph and hide a subset of edges
    edges = list(G.edges())
    hide_count = max(1, int(0.02 * len(edges)))
    E_hidden = set(rng.choice(len(edges), size=hide_count, replace=False))
    G_obs = G.copy()
    hidden_edges = set()
    for idx in E_hidden:
        u, v = edges[idx]
        if G_obs.has_edge(u, v):
            G_obs.remove_edge(u, v)
            hidden_edges.add((u, v) if u < v else (v, u))

    # score candidates on observed graph
    scored = _score_pairs(G_obs, metric=metric)
    scored.sort(key=lambda x: x[2], reverse=True)

    K = max(1, int(len(scored) * k_ratio))
    topK = set()
    for i in range(min(K, len(scored))):
        u, v, _ = scored[i]
        topK.add((u, v) if u < v else (v, u))

    hit = len(topK & hidden_edges)
    rec = hit / len(hidden_edges) if hidden_edges else 0.0
    return float(rec)

# ==== 3) Contextual Inference (Content Leakage) ============================

def contextual_precision(
    posts: pd.DataFrame,
    comments: pd.DataFrame,
    sensitive_labels: set = SENSITIVE_ENTS
) -> float:
    """
    Proxy "precision" of sensitive entity extraction on publicly visible content.
    Treat any detected sensitive entity as a 'positive'. Precision = TP / (TP + FP)
    Here we don't have ground truth labels, so we approximate:
    - Count detections on public posts/comments as 'positives' (TP surrogate).
    - Count detections on private/friends as 'shouldn't-be-exposed' (FP surrogate)
      because an attacker shouldn't see them.
    """
    nlp = _load_spacy()
    tp = 0
    fp = 0

    def count(text: str) -> int:
        ents = _extract_entities(nlp, text)
        return sum(1 for e in ents if e in sensitive_labels)

    for _, p in posts.iterrows():
        vis = str(p.get("visibility", "public")).lower()
        c = count(str(p.get("text", "")))
        if vis == "public":
            tp += c
        else:
            fp += c
    for _, cmt in comments.iterrows():
        vis = str(cmt.get("visibility", "public")).lower()
        c = count(str(cmt.get("text", "")))
        if vis == "public":
            tp += c
        else:
            fp += c

    denom = tp + fp
    return float(tp / denom) if denom else 0.0

# ==== Post-Recommendation Views ============================================

def apply_privacy_recommendations_to_views(
    attrs_df: pd.DataFrame,
    posts: pd.DataFrame,
    comments: pd.DataFrame,
    recs: pd.DataFrame,
    scrub_text: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Produce "after" views by:
    - Reducing visibility of suggested attributes to 'private'
    - Optionally scrub obvious sensitive markers in text (emails, numbers that look like phones)
    """
    attrs2 = attrs_df.copy()
    posts2 = posts.copy()
    comments2 = comments.copy()

    # 1) tighten attribute vis
    rec_map = {str(r.user_id): str(r.suggest).split(",") if isinstance(r.suggest, str) else [] for _, r in recs.iterrows()}
    attrs2 = attrs2.set_index("user_id")
    for uid, fields in rec_map.items():
        if uid not in attrs2.index:
            continue
        for a in fields:
            key = f"{a}_vis"
            if key in attrs2.columns:
                attrs2.at[uid, key] = "private"
    attrs2 = attrs2.reset_index()

    # 2) scrub content a bit
    if scrub_text:
        email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        phone_re = re.compile(r"\b\d{2,3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
        date_re = re.compile(r"\b(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b")

        def scrub(s: str) -> str:
            if not isinstance(s, str) or not s:
                return s
            s = email_re.sub("[EMAIL]", s)
            s = phone_re.sub("[PHONE]", s)
            s = date_re.sub("[DATE]", s)
            return s

        for df in (posts2, comments2):
            if "text" in df.columns:
                df["text"] = df["text"].apply(scrub)

        # optionally downgrade visibilities of risky posts
        for df in (posts2, comments2):
            if "visibility" in df.columns:
                df.loc[df["visibility"].str.lower().eq("public"), "visibility"] = "friends"

    return attrs2, posts2, comments2
