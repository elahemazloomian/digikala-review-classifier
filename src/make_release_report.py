"""Build release_report.json from the saved result files.

Nothing is retrained and no number is typed by hand except the short lists of
limitations and known issues at the bottom. Run from the project root:

    python src/make_release_report.py
"""

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
TEST = REPORTS / "test_results"
OUT = ROOT / "release_report.json"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lib_versions(names):
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "not installed"
    return out


def head_metrics(m):
    keep = ["n", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
            "mcc", "nll", "ece", "ordinal_distance_error", "share_distance_2_errors", "quadratic_weighted_kappa",
            "per_class", "confusion_matrix"]
    return {k: m[k] for k in keep if k in m}


def history_summary(name, path, kind):
    h = load(path)
    best = max(h["evals"], key=lambda e: e["mean_macro_f1"])
    args = h["args"]
    return {
        "name": name,
        "type": kind,
        "encoder": args.get("model_name"),
        "total_params": h["total_params"],
        "trainable_params": h["trainable_params"],
        "training_rows_argument": args.get("train_rows"),
        "epochs": args.get("epochs"),
        "max_length": args.get("max_length"),
        "seed": args.get("seed"),
        "best_step": best["step"],
        "best_validation_mean_macro_f1": best["mean_macro_f1"],
        "evaluations": len(h["evals"]),
        "training_minutes": h["minutes"],
        "peak_gpu_gb": h["peak_gpu_gb"],
        "history_file": str(path.relative_to(ROOT)).replace("\\", "/"),
    }


split = load(REPORTS / "split_manifest.json")
pseudo = load(REPORTS / "pseudo_selection_manifest.json")
cmp_ = load(TEST / "comparison.json")
val = load(REPORTS / "ft_200k_eval_validation.json")

models = [
    {
        "name": "tfidf_baseline",
        "type": "TF-IDF word 1-2 grams + logistic regression (two separate models)",
        "artifact_path": "artifacts/tfidf_word_{recommendation_status,rating_band}_{vectorizer,logreg}.joblib",
        "training_rows_note": "all train_supervised rows with a valid target (2,945,521 for recommendation, "
                              "3,266,387 for rating band); copied from project notes, not recomputed here",
    },
    {**history_summary("transformer_supervised", REPORTS / "ft_200k_history.json", "fine-tuned Transformer, two heads"),
     "artifact_path": "artifacts/ft_200k/best.pt"},
    {**history_summary("transformer_control", REPORTS / "ctrl_200k_history.json",
                       "control: continued training from ft_200k on labeled rows only"),
     "artifact_path": "artifacts/ctrl_200k/best.pt"},
    {**history_summary("transformer_ssl_v2", REPORTS / "ssl_200k_v2_history.json",
                       "semi-supervised v2: labeled + pseudo rows, class weights from labeled rows only"),
     "artifact_path": "artifacts/ssl_200k_v2/best.pt"},
    {**history_summary("transformer_ssl_v1_combined_weights", REPORTS / "ssl_200k_history.json",
                       "semi-supervised v1: class weights from labeled + pseudo rows (mistake, kept as sensitivity result)"),
     "artifact_path": "artifacts/ssl_200k/best.pt"},
]

test_models = list(cmp_["models"]["rec"].keys())
test = {"bootstrap_draws": cmp_["draws"], "bootstrap_seed": cmp_["seed"], "models": {}}
for name in test_models:
    f = TEST / f"{name}.json"
    t = load(f)
    entry = {"rows_requested": t.get("rows_requested"), "seed": t.get("seed"), "run_dir": str(t.get("run_dir", "")).replace("\\", "/")}
    for head in ("rec", "band"):
        entry[head] = head_metrics(t[head])
        entry[head]["bootstrap_95ci"] = {k: {"point": v[0], "low": v[1], "high": v[2]} for k, v in cmp_["models"][head][name].items()}
    test["models"][name] = entry

paired = {}
for head in ("rec", "band"):
    paired[head] = {}
    for pair, metrics in cmp_["pairs"][head].items():
        paired[head][pair] = {k: {"difference": v[0], "low": v[1], "high": v[2],
                                  "interval_excludes_zero": bool(v[1] > 0 or v[2] < 0)} for k, v in metrics.items()}

report = {
    "project": "digikala-review-classifier",
    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    "environment_of_this_report": {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": lib_versions(["duckdb", "pandas", "numpy", "pyarrow", "scikit-learn", "joblib", "psutil",
                                   "matplotlib", "torch", "transformers", "tokenizers"]),
    },
    "environment_of_training": "trained on Windows with torch 2.14.1+cu132 (CUDA, 4 GB GPU); notebooks and offline "
                               "prediction tested on Ubuntu with the CPU build of torch (see requirements.txt)",
    "seeds": {
        "split_seed": split["split_seed"],
        "mask_seed": split["mask_seed"],
        "mask_rate": split["mask_rate"],
        "training_subset_seed": models[1]["seed"],
        "pseudo_label_cap_seed": pseudo.get("seed"),
        "bootstrap_seed": cmp_["seed"],
    },
    "data": {
        "clean_rows_total": split["clean_rows_total"],
        "products_per_split": split["products_per_split"],
        "rows_per_file": split["rows_per_file"],
        "split_strategy": split["split_strategy"],
        "rating_bands": split["rating_bands"],
        "masking": split["masking"],
    },
    "models": models,
    "metrics": {
        "validation_selected_checkpoint": {
            "source": "reports/ft_200k_eval_validation.json",
            "rows_requested": val.get("rows_requested"),
            "recommendation": head_metrics(val["rec"]),
            "rating_band": head_metrics(val["band"]),
        },
        "test": test,
        "paired_differences": paired,
    },
    "semi_supervised": {
        "threshold": pseudo.get("threshold"),
        "pool_rows": pseudo.get("pool_rows"),
        "accepted_before_cap": pseudo.get("accepted_before_cap"),
        "accepted_share_of_pool": pseudo.get("accepted_share_of_pool"),
        "per_class_before_cap": pseudo.get("per_class_before_cap"),
        "kept_rows": pseudo.get("kept_rows"),
        "per_class_kept": pseudo.get("per_class_kept"),
        "mean_confidence_kept": pseudo.get("mean_confidence_kept"),
        "note": pseudo.get("note"),
        "runs": ["transformer_control", "transformer_ssl_v2", "transformer_ssl_v1_combined_weights"],
    },
    "limitations": [
        "Transformer fine-tuned on 200,000 training rows (about 6% of the labeled rows), word-embedding table frozen, max_length 64",
        "one training run per setting; bootstrap intervals describe test-sampling noise only",
        "the recommendation label is partly subjective: the same text appears with different labels",
        "the unlabeled pool is artificial (hidden labels of training rows)",
        "colloquial negation and hedged phrases are still misclassified in some cases",
    ],
    "known_issues": [
        "semi-supervised v1 used class weights computed on labeled and pseudo rows together; kept only as a sensitivity result",
        "the second human review of the error categories is pending",
        "331,599 products were counted in the audit and 331,597 remain after cleaning; the difference is not verified",
        "the artificial unlabeled file contains body_norm in addition to id and body",
        "the checkpoint was trained with a different torch version than the one used for the offline demo",
    ],
}

OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print("written:", OUT, "| size:", OUT.stat().st_size, "bytes")
