"""TF-IDF (word 1-2 grams) + logistic regression baseline for both heads.

Run from the project root:
    python src/train_tfidf_baseline.py
Optional:
    python src/train_tfidf_baseline.py --max-train-rows 1000000 --max-iter 300
Only the training file is used for fitting; the validation file is used for scores.
The test file is never read here.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
SEED = 2026

HEADS = {
    "recommendation_status": ["no_idea", "not_recommended", "recommended"],
    "rating_band": ["low", "medium", "high"],
}
BAND_CODE = {"low": 0, "medium": 1, "high": 2}


def mem_gb():
    return round(psutil.Process().memory_info().rss / 1024 ** 3, 2)


def load_rows(con, file_name, target, max_rows):
    path = (PROCESSED / file_name).as_posix()
    sample = f"USING SAMPLE reservoir({max_rows} ROWS) REPEATABLE ({SEED})" if max_rows else ""
    return con.execute(f"""
        SELECT body_norm, {target} AS y
        FROM read_csv('{path}', header=true, all_varchar=true)
        WHERE {target} IS NOT NULL AND body_norm IS NOT NULL
        {sample}
    """).df()


def run_head(con, target, labels, max_train_rows, max_iter):
    print(f"\n===== head: {target} =====", flush=True)
    t_all = time.time()
    train = load_rows(con, "train_supervised.csv", target, max_train_rows)
    val = load_rows(con, "validation.csv", target, None)
    print("train rows:", len(train), "| validation rows:", len(val), "| memory:", mem_gb(), "GB", flush=True)

    t0 = time.time()
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                          max_features=300000, sublinear_tf=True,
                          token_pattern=r"(?u)\b\w+\b")
    X_tr = vec.fit_transform(train["body_norm"])
    X_va = vec.transform(val["body_norm"])
    print("matrix:", X_tr.shape, "seconds:", round(time.time() - t0, 1), "| memory:", mem_gb(), "GB", flush=True)

    t0 = time.time()
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=max_iter)
    clf.fit(X_tr, train["y"])
    n_iter = int(clf.n_iter_[0])
    print("fit seconds:", round(time.time() - t0, 1), "| iterations used:", n_iter,
          "of", max_iter, "| memory:", mem_gb(), "GB", flush=True)

    pred = clf.predict(X_va)
    f1_each = f1_score(val["y"], pred, labels=labels, average=None)
    metrics = {
        "head": target,
        "train_rows": int(len(train)), "validation_rows": int(len(val)),
        "n_features": int(X_tr.shape[1]), "iterations_used": n_iter, "max_iter": max_iter,
        "converged": n_iter < max_iter,
        "macro_f1": round(float(f1_score(val["y"], pred, average="macro")), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(val["y"], pred)), 4),
        "f1_per_class": {lab: round(float(v), 4) for lab, v in zip(labels, f1_each)},
        "confusion_matrix": {"labels": labels,
                             "rows_true_cols_pred": confusion_matrix(val["y"], pred, labels=labels).tolist()},
    }
    if target == "rating_band":
        dist = (val["y"].map(BAND_CODE) - pd.Series(pred, index=val.index).map(BAND_CODE)).abs()
        metrics["ordinal_distance_error"] = round(float(dist.mean()), 4)
        metrics["share_distance_2_errors"] = round(float((dist == 2).mean()), 4)
    print(classification_report(val["y"], pred, labels=labels, digits=3), flush=True)
    print("macro F1:", metrics["macro_f1"], "| balanced acc:", metrics["balanced_accuracy"], flush=True)

    # the saved vectorizer does not need the list of dropped words
    if hasattr(vec, "stop_words_"):
        vec.stop_words_ = None
    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(vec, ARTIFACTS / f"tfidf_word_{target}_vectorizer.joblib")
    joblib.dump(clf, ARTIFACTS / f"tfidf_word_{target}_logreg.joblib")
    with open(REPORTS / f"baseline_tfidf_word_{target}_validation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("head finished in", round((time.time() - t_all) / 60, 1), "minutes", flush=True)
    del train, val, X_tr, X_va, vec, clf
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    con = duckdb.connect()
    for target, labels in HEADS.items():
        run_head(con, target, labels, args.max_train_rows, args.max_iter)
    print("\nAll done.")


if __name__ == "__main__":
    main()