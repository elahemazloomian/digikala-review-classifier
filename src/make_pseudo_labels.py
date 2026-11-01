"""Select confident pseudo-labels from the saved pool probabilities (recommendation head only).

Run from the project root, for example:
    python src/make_pseudo_labels.py --run-dir artifacts/ft_200k --threshold 0.90 --max-rows 200000
Reads : artifacts/<run>/pseudo_probs/part_*.parquet  and  data/processed/train_pseudo_unlabeled.csv
Writes: data/processed/pseudo_labeled_accepted.csv   and   reports/pseudo_selection_manifest.json
The true (hidden) labels are NOT read here. They are only used afterwards, to audit quality.
"""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
LABELS = ["no_idea", "not_recommended", "recommended"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--max-rows", type=int, default=None, help="keep at most this many accepted rows (random, seeded)")
    ap.add_argument("--pool", default="train_pseudo_unlabeled.csv")
    ap.add_argument("--out", default="pseudo_labeled_accepted.csv")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    parts = sorted((ROOT / args.run_dir / "pseudo_probs").glob("part_*.parquet"))
    probs = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
    P = probs[[f"p_rec_{l}" for l in LABELS]].to_numpy()
    probs["pred"] = np.array(LABELS)[P.argmax(1)]
    probs["conf"] = P.max(1)
    print("pool rows:", len(probs), "| unique ids:", probs["id"].nunique(), flush=True)

    accepted = probs[probs["conf"] >= args.threshold]
    before = accepted["pred"].value_counts().reindex(LABELS, fill_value=0)
    if args.max_rows and len(accepted) > args.max_rows:
        accepted = accepted.sample(n=args.max_rows, random_state=args.seed)
    after = accepted["pred"].value_counts().reindex(LABELS, fill_value=0)
    print(f"accepted at threshold {args.threshold}: {int(before.sum())} "
          f"({100 * before.sum() / len(probs):.1f}% of the pool) | kept after cap: {len(accepted)}", flush=True)
    print("per predicted class (before cap):", before.to_dict(), flush=True)
    print("per predicted class (kept)      :", after.to_dict(), flush=True)

    con = duckdb.connect()
    con.register("acc", accepted[["id", "pred"]])
    pool = (PROCESSED / args.pool).as_posix()
    out = (PROCESSED / args.out).as_posix()
    con.execute(f"""
        COPY (
          SELECT p.id, p.body, p.body_norm, a.pred AS recommendation_status,
                 NULL::VARCHAR AS rating_band, NULL::VARCHAR AS rate
          FROM read_csv('{pool}', header=true, all_varchar=true) p
          JOIN acc a ON p.id = a.id
        ) TO '{out}' (HEADER, DELIMITER ',')
    """)
    n_out = con.execute(f"SELECT COUNT(*) FROM read_csv('{out}', header=true, all_varchar=true)").fetchall()[0][0]
    print("rows written:", n_out, "->", out, flush=True)
    assert n_out == len(accepted), "some accepted ids were not found in the pool file"

    manifest = {"run_dir": args.run_dir, "threshold": args.threshold, "max_rows": args.max_rows,
                "seed": args.seed, "pool_rows": int(len(probs)), "accepted_before_cap": int(before.sum()),
                "accepted_share_of_pool": round(float(before.sum() / len(probs)), 4),
                "per_class_before_cap": {k: int(v) for k, v in before.items()},
                "kept_rows": int(len(accepted)), "per_class_kept": {k: int(v) for k, v in after.items()},
                "mean_confidence_kept": round(float(accepted["conf"].mean()), 4),
                "note": "selection uses model probabilities only; hidden labels were not read"}
    (ROOT / "reports").mkdir(exist_ok=True)
    with open(ROOT / "reports" / "pseudo_selection_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("manifest saved.", flush=True)


if __name__ == "__main__":
    main()
