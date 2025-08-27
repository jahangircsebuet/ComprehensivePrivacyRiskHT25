import os
import argparse
import pandas as pd
import networkx as nx

from .data_io import load_snap_graph, load_text_data, generate_synthetic_attributes, ensure_output_dir
from .recommend import recommend_privacy_changes
from .attacks import (
    attribute_inference_accuracy,
    link_prediction_recall,
    contextual_precision,
    apply_privacy_recommendations_to_views
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap_graph", required=True)
    ap.add_argument("--koo_posts", required=True)
    ap.add_argument("--koo_comments", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--hidden_attr", default="relationship")
    ap.add_argument("--hide_rate", type=float, default=0.5)
    ap.add_argument("--lp_metric", choices=["jaccard", "adamic_adar"], default="jaccard")
    ap.add_argument("--k_ratio", type=float, default=0.01)
    ap.add_argument("--scrub_text", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_output_dir(args.outdir)

    # Load data
    G = load_snap_graph(args.snap_graph)
    posts, comments = load_text_data(args.koo_posts, args.koo_comments)
    attrs = generate_synthetic_attributes(G)

    # --- BEFORE ---
    att_acc_b = attribute_inference_accuracy(G, attrs, hidden_attr=args.hidden_attr, hide_rate=args.hide_rate)
    lp_rec_b  = link_prediction_recall(G, k_ratio=args.k_ratio, metric=args.lp_metric)
    ctx_prec_b = contextual_precision(posts, comments)

    # Recommendations (from the CPRS pipeline’s heuristic)
    # We don’t need CPRS values here—use top quantile suggestion logic on synthetic attrs
    # Minimal stub: pretend all users are potentially risky; generate a few field tightenings
    # Better: reuse the existing recommend.py but it expects CPRS CSV; we can simulate using a fake score
    fake_scores = pd.DataFrame({"CPRS": 0.0}, index=attrs["user_id"].astype(str))
    fake_scores.index.name = "user_id"
    recs = recommend_privacy_changes(attrs, fake_scores.reset_index())
    recs_path = os.path.join(args.outdir, "attack_recommendations.csv")
    recs.to_csv(recs_path, index=False)

    # --- AFTER (apply recs, scrub if requested) ---
    attrs2, posts2, comments2 = apply_privacy_recommendations_to_views(attrs, posts, comments, recs, scrub_text=args.scrub_text)

    att_acc_a = attribute_inference_accuracy(G, attrs2, hidden_attr=args.hidden_attr, hide_rate=args.hide_rate)
    lp_rec_a  = link_prediction_recall(G, k_ratio=args.k_ratio, metric=args.lp_metric)
    ctx_prec_a = contextual_precision(posts2, comments2)

    # Privacy gains
    def gain(before, after):
        return 0.0 if before == 0 else (before - after) / before * 100.0

    rows = [
        ["Attribute Inference (acc)", att_acc_b, att_acc_a, gain(att_acc_b, att_acc_a)],
        [f"Link Prediction@{args.k_ratio} ({args.lp_metric})", lp_rec_b, lp_rec_a, gain(lp_rec_b, lp_rec_a)],
        ["Contextual Inference (precision)", ctx_prec_b, ctx_prec_a, gain(ctx_prec_b, ctx_prec_a)],
    ]
    out = pd.DataFrame(rows, columns=["Attack", "Before", "After", "Privacy Gain (%)"])
    out_path = os.path.join(args.outdir, "attack_simulation_results.csv")
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))
    print(f"\n[OK] Saved results to {out_path}")
    print(f"[OK] Saved recommendations to {recs_path}")

if __name__ == "__main__":
    main()
