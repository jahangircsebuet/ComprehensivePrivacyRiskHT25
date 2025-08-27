# CPRS: Comprehensive Privacy Risk Scoring

Implements APRS, SGPRS (SimRank + PageRank), CBPRS (NER + visibility), AHP weighting, and overall CPRS.

## Quickstart

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python -m cprs.main \
  --snap_graph path/to/facebook_combined.txt \
  --koo_posts path/to/posts.csv \
  --koo_comments path/to/comments.csv \
  --platform content  # or graph / equal
