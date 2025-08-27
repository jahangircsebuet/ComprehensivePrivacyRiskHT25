import argparse
import os
import pandas as pd
from .data_io import load_snap_graph, load_text_data, generate_synthetic_attributes, ensure_output_dir, friends_count_map
from .aprs import compute_aprs
from .sgprs import compute_sgprs
from .cbprs import compute_cbprs
from .ahp import ahp_weights_content, ahp_weights_graph, equal_weights
from .cprs import compute_cprs
from .recommend import recommend_privacy_changes

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap_graph", required=True, help="Path to SNAP-like edge list")
    ap.add_argument("--koo_posts", required=True, help="CSV with columns: user_id,text,visibility")
    ap.add_argument("--koo_comments", required=True, help="CSV with columns: user_id,text,visibility")
    ap.add_argument("--platform", choices=["equal", "content", "graph"], default="equal")
    ap.add_argument("--outdir", default="outputs")
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_output_dir(args.outdir)

    # 1) Load data
    G = load_snap_graph(args.snap_graph)
    posts, comments = load_text_data(args.koo_posts, args.koo_comments)

    # 2) Synthetic attributes (since SNAP graph has no attributes)
    attrs = generate_synthetic_attributes(G)

    # 3) Helpers
    fcount = friends_count_map(G)
    amax = max(fcount.values()) if len(fcount) else 1

    # 4) Component scores
    aprs = compute_aprs(attrs, fcount, amax)             # Series index: user_id
    sgprs = compute_sgprs(G, aprs)                        # Series index: node (user_id)
    cbprs = compute_cbprs(posts, comments, fcount, amax) # Series index: user_id

    # 5) Weights
    if args.platform == "content":
        w_aprs, w_sgprs, w_cbprs = ahp_weights_content()
    elif args.platform == "graph":
        w_aprs, w_sgprs, w_cbprs = ahp_weights_graph()
    else:
        w_aprs, w_sgprs, w_cbprs = equal_weights()

    # 6) CPRS
    df_scores = compute_cprs(aprs, sgprs, cbprs, w_aprs, w_sgprs, w_cbprs)
    df_scores.index.name = "user_id"
    df_scores.to_csv(os.path.join(args.outdir, "user_scores.csv"))

    # 7) Basic recommendations
    recs = recommend_privacy_changes(attrs, df_scores)
    recs.to_csv(os.path.join(args.outdir, "recommendations.csv"), index=False)

    print(f"[OK] Saved {len(df_scores)} user scores and {len(recs)} recommendations in {args.outdir}")

if __name__ == "__main__":
    main()
