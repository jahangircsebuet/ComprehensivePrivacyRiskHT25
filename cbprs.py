import pandas as pd
from typing import Dict
from collections import Counter
from .utils import VIS_MAP, safe_div

# Simple entity sensitivity weights (tweak as needed)
ENTITY_WEIGHTS = {
    "PERSON": 1.0,
    "GPE": 0.9,    # country/city/state
    "LOC": 0.8,
    "ORG": 0.7,
    "DATE": 0.6,
    "TIME": 0.5,
    "MONEY": 0.9,
    "EMAIL": 1.0,  # custom handling
}

def _load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None  # allow pipeline to run without NER (score=0)

def _extract_entities(nlp, text: str) -> Counter:
    ents = Counter()
    if not nlp or not text:
        return ents
    doc = nlp(text)
    for e in doc.ents:
        ents[e.label_] += 1
    # crude email catch
    if "@" in text:
        ents["EMAIL"] += text.count("@")
    return ents

def _visibility_value(vis: str, friends: int, amax: int) -> float:
    v = VIS_MAP.get(vis, 1.0)
    if v is None:
        return safe_div(friends, amax, default=0.0)
    return v

def _score_text(nlp, text: str) -> float:
    ents = _extract_entities(nlp, text)
    return sum(ENTITY_WEIGHTS.get(k, 0.3) * c for k, c in ents.items())

def compute_cbprs(
    posts: pd.DataFrame,
    comments: pd.DataFrame,
    friends_count: Dict[str, int],
    amax: int
) -> pd.Series:
    """
    For each user: sum over posts: [ (post_entity_score * V(post)) + sum_k (comment_entity_score_k * V(post)) ]
    """
    nlp = _load_spacy()
    # ensure columns
    posts = posts.copy()
    comments = comments.copy()
    if "visibility" not in posts.columns:
        posts["visibility"] = "public"

    post_scores = []
    for _, p in posts.iterrows():
        uid = str(p["user_id"])
        vis = p.get("visibility", "public")
        vval = _visibility_value(vis, friends_count.get(uid, 0), amax)
        p_score = _score_text(nlp, str(p.get("text", "")))
        # comments attached to this post? If you have post_id, join; otherwise approximate by author
        c_sum = 0.0
        # If you have a post_id column, replace with comments[comments.post_id == p.post_id]
        cands = comments[comments["user_id"] == p["user_id"]]
        for _, c in cands.iterrows():
            c_sum += _score_text(nlp, str(c.get("text", ""))) * vval
        total = p_score * vval + c_sum
        post_scores.append((uid, total))

    # Aggregate per user
    agg = {}
    for uid, sc in post_scores:
        agg[uid] = agg.get(uid, 0.0) + sc
    cbprs = pd.Series(agg, name="CBPRS")
    return cbprs
