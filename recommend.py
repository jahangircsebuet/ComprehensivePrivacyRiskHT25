import pandas as pd
from .data_io import ATTRIBUTES

def recommend_privacy_changes(user_attr_df: pd.DataFrame, user_scores: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """
    Very simple heuristic recommender:
    - For users in top CPRS quantile, suggest tightening the top_k highest-sensitivity attributes currently public/friends to more private.
    """
    q90 = user_scores["CPRS"].quantile(0.90)
    risky = user_scores[user_scores["CPRS"] >= q90].index.astype(str).tolist()

    # For simplicity, rank attributes by how often they're public and known sensitive ones
    SENSITIVE = ["email", "dob", "workplace", "from_location", "lives_in", "mobile"]
    rows = []
    df = user_attr_df.set_index("user_id")
    for u in risky:
        if u not in df.index: 
            continue
        row = df.loc[u]
        candidates = []
        for a in SENSITIVE:
            vis = row.get(f"{a}_vis", "public")
            if vis in ("public", "friends"):
                candidates.append(a)
        suggest = candidates[:top_k]
        rows.append({"user_id": u, "suggest": ",".join(suggest), "action": "reduce_visibility"})
    return pd.DataFrame(rows)
