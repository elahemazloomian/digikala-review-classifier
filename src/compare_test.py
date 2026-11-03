"""Paired bootstrap comparison of the models evaluated on the TEST sample.

Run from the project root, after evaluate_test.py has been run for every model:
    python src/compare_test.py
Reads reports/test_results/<name>_predictions.parquet, writes reports/test_results/comparison.json
Every bootstrap draw resamples the SAME test rows for all models, so differences are paired.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "test_results"
NAMES = ["tfidf_baseline", "transformer_supervised", "transformer_control",
         "transformer_ssl_v2", "transformer_ssl_v1_combined_weights"]
PAIRS = [("transformer_supervised", "tfidf_baseline"),
         ("transformer_control", "transformer_supervised"),
         ("transformer_ssl_v2", "transformer_control"),
         ("transformer_ssl_v2", "transformer_supervised"),
         ("transformer_ssl_v1_combined_weights", "transformer_control")]
HEADS = {"rec": ["no_idea", "not_recommended", "recommended"], "band": ["low", "medium", "high"]}
METRICS = ["macro_f1", "accuracy", "balanced_accuracy", "ordinal_error"]


def metrics_from_conf(C, ordinal):
    tp = np.diag(C).astype(float)
    support, predicted = C.sum(1).astype(float), C.sum(0).astype(float)
    prec = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    k = C.shape[0]
    err = np.nan
    if ordinal:
        i, j = np.indices((k, k))
        err = float((np.abs(i - j) * C).sum() / C.sum())
    return [f1.mean(), tp.sum() / C.sum(), rec.mean(), err]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    data = {n: pd.read_parquet(RES / f"{n}_predictions.parquet") for n in NAMES}
    ids = data[NAMES[0]]["id"].tolist()
    for n in NAMES[1:]:
        assert data[n]["id"].tolist() == ids, f"{n}: test rows differ from {NAMES[0]}"
    rng = np.random.default_rng(args.seed)
    out = {"draws": args.draws, "seed": args.seed, "models": {}, "pairs": {}}

    for head, labels in HEADS.items():
        k, ordinal = len(labels), head == "band"
        true = data[NAMES[0]][f"{head}_true"].to_numpy()
        keep = true >= 0
        true = true[keep]
        pred = {n: data[n].loc[keep, [f"p_{head}_{l}" for l in labels]].to_numpy().argmax(1) for n in NAMES}
        n_rows = len(true)
        boot = np.zeros((args.draws, len(NAMES), len(METRICS)))
        for b in range(args.draws):
            idx = rng.integers(0, n_rows, n_rows)
            for mi, n in enumerate(NAMES):
                C = np.bincount(true[idx] * k + pred[n][idx], minlength=k * k).reshape(k, k)
                boot[b, mi] = metrics_from_conf(C, ordinal)
        point = {n: metrics_from_conf(np.bincount(true * k + pred[n], minlength=k * k).reshape(k, k), ordinal)
                 for n in NAMES}
        print(f"\n===== {head} ({n_rows:,} rows, {args.draws} paired bootstrap draws) =====")
        for mi, n in enumerate(NAMES):
            line = f"{n:38s}"
            for m_i, m in enumerate(METRICS):
                if m == "ordinal_error" and not ordinal:
                    continue
                lo, hi = np.percentile(boot[:, mi, m_i], [2.5, 97.5])
                line += f" | {m} {point[n][m_i]:.4f} [{lo:.4f}, {hi:.4f}]"
                out["models"].setdefault(head, {}).setdefault(n, {})[m] = [point[n][m_i], float(lo), float(hi)]
            print(line)
        print("\n  differences (first minus second); 95% interval; share of draws where first is better")
        for a, b_ in PAIRS:
            ia, ib = NAMES.index(a), NAMES.index(b_)
            for m_i, m in enumerate(METRICS):
                if m == "ordinal_error" and not ordinal:
                    continue
                d = boot[:, ia, m_i] - boot[:, ib, m_i]
                lo, hi = np.percentile(d, [2.5, 97.5])
                better = float((d < 0).mean() if m == "ordinal_error" else (d > 0).mean())
                est = point[a][m_i] - point[b_]  [m_i]
                flag = "  <- interval excludes 0" if (lo > 0 or hi < 0) else ""
                print(f"  {a} - {b_} | {m}: {est:+.4f} [{lo:+.4f}, {hi:+.4f}] | better in {100 * better:.1f}%{flag}")
                out["pairs"].setdefault(head, {}).setdefault(f"{a} - {b_}", {})[m] = [est, float(lo), float(hi), better]
    with open(RES / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved:", RES / "comparison.json")


if __name__ == "__main__":
    main()
