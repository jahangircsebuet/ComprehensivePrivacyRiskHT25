import os
import random
import pandas as pd
import networkx as nx
from typing import Tuple, Dict

ATTRIBUTES = [
    "mobile", "email", "gender", "pronoun", "dob", "relationship",
    "from_location", "lives_in", "school", "workplace"
]

def load_snap_graph(path: str) -> nx.Graph:
    # nx.read_edgelist(path) reads each line, splits on whitespace, and creates an undirected edge.
    # nodetype=str forces node IDs to be stored as strings (important for consistent joins with CSVs later).
    # Returns a networkx.Graph object with .nodes(), .edges(), .degree(), etc.
    
    """Edge-list loader for SNAP Facebook ego networks (or similar)."""
    G = nx.read_edgelist(path, nodetype=str)
    return G

def load_text_data(posts_csv: str, comments_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    posts = pd.read_csv(posts_csv)
    comments = pd.read_csv(comments_csv)
    # expected columns: user_id, text, visibility
    for df in (posts, comments):
        if "visibility" not in df.columns:
            df["visibility"] = "public"
        if "user_id" not in df.columns:
            raise ValueError("posts/comments must include user_id")
        if "text" not in df.columns:
            df["text"] = ""
    return posts, comments

def generate_synthetic_attributes(G: nx.Graph, seed: int = 42) -> pd.DataFrame:
    """Create synthetic user attributes & privacy settings with plausible distributions."""
    random.seed(seed)
    rows = []
    for u in G.nodes():
        # simple plausible values
        row = {
            "user_id": u,
            "mobile": f"{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            "email": f"user{random.randint(1, 500000)}@example.com",
            "gender": random.choice(["M", "F", "Non-binary"]),
            "pronoun": random.choice(["he/him", "she/her", "they/them"]),
            "dob": f"{random.randint(1970, 2010)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "relationship": random.choice(["single", "in_a_relationship", "married"]),
            "from_location": random.choice(["CA", "NY", "TX", "ND", "CT", "WA"]),
            "lives_in": random.choice(["CA", "NY", "TX", "ND", "CT", "WA"]),
            "school": random.choice(["Harvard", "MIT", "UCLA", "UIUC", "Stanford"]),
            "workplace": random.choice(["Microsoft", "Google", "OpenAI", "Meta", "Amazon"])
        }
        # visibility skew: friends-heavy, small public/private tails
        vis_map = {}
        for a in ATTRIBUTES:
            r = random.random()
            if r < 0.20: vis_map[a] = "public"
            elif r < 0.78: vis_map[a] = "friends"
            else: vis_map[a] = "private"
        row.update({f"{a}_vis": vis_map[a] for a in ATTRIBUTES})
        rows.append(row)
    return pd.DataFrame(rows)

def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def friends_count_map(G: nx.Graph) -> Dict[str, int]:
    return {n: G.degree(n) for n in G.nodes()}
