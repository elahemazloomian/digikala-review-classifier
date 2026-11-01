"""FINAL evaluation on the TEST file. Run each finished model ONCE.

Examples (run from the project root):
    python src/evaluate_test.py --model tfidf --name tfidf_baseline
    python src/evaluate_test.py --model transformer --run-dir artifacts/ft_200k --name transformer_supervised
All models use the same random test sample (same seed and size), so they can be compared row by row.
A second run with the same --name is refused unless you add --allow-repeat (to avoid repeated peeking).
Writes reports/test_results/<name>.json and <name>_predictions.parquet
"""
import argparse
import json
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from finetune_transformer import (BAND_ID, BAND_LABELS, IGNORE, PROCESSED, REC_ID, REC_LABELS, ROOT,
                                  ReviewClassifier, head_metrics, make_collate)
from predict_pseudo import TextOnly

ARTIFACTS = ROOT / "artifacts"


def load_test(rows, seed):
    path = (PROCESSED / "test.csv").as_posix()
    sample = f"USING SAMPLE reservoir({rows} ROWS) REPEATABLE ({seed})" if rows else ""
    df = duckdb.connect().execute(f"""
        SELECT id, body_norm, recommendation_status AS rec, rating_band AS band
        FROM read_csv('{path}', header=true, all_varchar=true)
        WHERE body_norm IS NOT NULL
          AND (recommendation_status IS NOT NULL OR rating_band IS NOT NULL)
        {sample}
    """).df()
    df = df.sort_values("id", kind="stable").reset_index(drop=True)  # same row order for every model
    df["rec_id"] = df["rec"].map(REC_ID).fillna(IGNORE).astype(int)
    df["band_id"] = df["band"].map(BAND_ID).fillna(IGNORE).astype(int)
    return df


def predict_transformer(run_dir, texts, batch_size):
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(run_dir)
    model = ReviewClassifier(ckpt["model_name"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    order = np.argsort([len(t) for t in texts], kind="stable")  # similar lengths together: less padding
    sorted_texts = [texts[i] for i in order]
    loader = DataLoader(TextOnly(sorted_texts), batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate(tokenizer, ckpt["max_length"]))
    rec_p, band_p = [], []
    with torch.no_grad():
        for enc, _, _ in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                lr_, lb_ = model(enc)
            rec_p.append(torch.softmax(lr_.float(), 1).cpu().numpy())
            band_p.append(torch.softmax(lb_.float(), 1).cpu().numpy())
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return np.concatenate(rec_p)[inv], np.concatenate(band_p)[inv]


def predict_tfidf(texts):
    out = []
    for head, labels in (("recommendation_status", REC_LABELS), ("rating_band", BAND_LABELS)):
        vec = joblib.load(ARTIFACTS / f"tfidf_word_{head}_vectorizer.joblib")
        clf = joblib.load(ARTIFACTS / f"tfidf_word_{head}_logreg.joblib")
        proba = clf.predict_proba(vec.transform(texts))
        cols = [list(clf.classes_).index(lab) for lab in labels]  # put columns in our label order
        out.append(proba[:, cols])
    return out[0], out[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tfidf", "transformer"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--rows", type=int, default=100000, help="size of the random test sample (0 = every row)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--allow-repeat", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "reports" / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.name}.json"
    if out_json.exists() and not args.allow_repeat:
        sys.exit(f"{out_json} already exists. The test set is evaluated once per model. "
                 f"Use --allow-repeat only if you really mean it.")
    if args.model == "transformer" and not args.run_dir:
        sys.exit("--run-dir is required for --model transformer")

    df = load_test(args.rows or None, args.seed)
    print("TEST rows:", len(df), "| model:", args.model, "| name:", args.name, flush=True)
    texts = df["body_norm"].tolist()
    if args.model == "tfidf":
        rec_p, band_p = predict_tfidf(texts)
    else:
        rec_p, band_p = predict_transformer(ROOT / args.run_dir, texts, args.batch_size)

    res = {"name": args.name, "model": args.model, "run_dir": args.run_dir, "rows_requested": args.rows,
           "seed": args.seed}
    for key, labels, ordinal, probs in (("rec", REC_LABELS, False, rec_p), ("band", BAND_LABELS, True, band_p)):
        y = df[f"{key}_id"].to_numpy()
        m = y != IGNORE
        res[key] = head_metrics(y[m], probs[m].argmax(1), probs[m], labels, ordinal)
        r = res[key]
        print(f"\n== {key} ({r['n']:,} rows) ==")
        print(f"accuracy {r['accuracy']:.4f} | balanced acc {r['balanced_accuracy']:.4f} | macro P {r['macro_precision']:.4f} "
              f"| macro R {r['macro_recall']:.4f} | macro F1 {r['macro_f1']:.4f} | ECE {r['ece']:.4f}")
        if ordinal:
            print(f"ordinal distance error {r['ordinal_distance_error']:.4f} | QWK {r['quadratic_weighted_kappa']:.4f}")
        for lab in labels:
            c = r["per_class"][lab]
            print(f"  {lab:16s} P {c['precision']:.3f}  R {c['recall']:.3f}  F1 {c['f1']:.3f}  n {c['support']:,}")
    pred = pd.DataFrame({"id": df["id"], "rec_true": df["rec_id"], "band_true": df["band_id"]})
    for i, lab in enumerate(REC_LABELS):
        pred[f"p_rec_{lab}"] = rec_p[:, i].astype("float32")
    for i, lab in enumerate(BAND_LABELS):
        pred[f"p_band_{lab}"] = band_p[:, i].astype("float32")
    pred.to_parquet(out_dir / f"{args.name}_predictions.parquet")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\nsaved:", out_json)


if __name__ == "__main__":
    main()
