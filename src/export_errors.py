"""Error analysis helper: error rates per text property + a stratified sample of mistakes to read.

Run from the project root, after evaluate_test.py has been run for the models:
    python src/export_errors.py
Reads : reports/test_results/<model>_predictions.parquet  and  data/processed/test.csv
Writes: reports/error_analysis/error_rates_by_flag.csv   (how often each kind of text is misclassified)
        reports/error_analysis/review_sample.csv         (about 80 mistakes with text, to read by hand)
The flags are simple automatic rules (word lists, lengths). They only help to find candidates;
the final category of every analysed mistake is decided by reading it.
Nothing here changes a model. The test file is used only to describe errors AFTER the final evaluation.
"""
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "test_results"
OUT = ROOT / "reports" / "error_analysis"
SEED = 2026
MAIN, BASE, SSL = "transformer_supervised", "tfidf_baseline", "transformer_ssl_v2"
REC = ["no_idea", "not_recommended", "recommended"]
BAND = ["low", "medium", "high"]

NEG_WORDS = {"نه", "نیست", "نبود", "نبودن", "نیستش", "نکن", "نکنید", "نخرید", "نخر", "ندارد", "نداره", "ندارم",
             "بدون", "هیچ", "نشد", "نداشت", "نگیرید", "نمیشه", "نمیخرم", "نمیارزه"}
CONTRAST = {"ولی", "اما", "ولیکن", "اگرچه", "هرچند", "البته", "متاسفانه", "متأسفانه", "منتها", "فقط"}
POS_WORDS = {"عالی", "عالیه", "خوب", "پیشنهاد", "راضی", "ممنون", "باکیفیت", "عالیست", "عالیم"}
NEG_WORDS_STRONG = {"بد", "افتضاح", "ضعیف", "خراب", "بیکیفیت", "پشیمون", "تقلبی", "مزخرف", "کلاهبرداری", "نخرید"}


def tokens(text):
    return [t.replace("\u200c", "") for t in str(text).split()]


def flags_for(row):
    t = tokens(row["body_norm"])
    ts = set(t)
    has_neg_form = any(w in NEG_WORDS for w in t) or any(w.startswith("نمی") for w in t)
    pos, neg = bool(ts & POS_WORDS), bool(ts & NEG_WORDS_STRONG)
    f = {
        "very_short_1_word": len(t) == 1,
        "short_le_3_words": len(t) <= 3,
        "long_ge_40_words": len(t) >= 40,
        "negation": has_neg_form,
        "contrast_mixed": bool(ts & CONTRAST),
        "elongated_letters": bool(re.search(r"(.)\1{2,}", str(row["body"]))),
        "arabic_letter_variants": bool(re.search("[يك]", str(row["body"]))),
        "possible_contradiction_label": (pos and not neg and not has_neg_form and row["rec_true_name"] == "not_recommended")
                                        or (neg and not pos and row["rec_true_name"] == "recommended"),
    }
    rate = row["rate"]
    f["band_boundary_rate"] = bool(pd.notna(rate) and (abs(rate - 2.5) <= 0.25 or abs(rate - 4.0) <= 0.25))
    return f


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = {n: pd.read_parquet(RES / f"{n}_predictions.parquet") for n in (MAIN, BASE, SSL)
            if (RES / f"{n}_predictions.parquet").exists()}
    assert MAIN in pred and BASE in pred, "need the predictions of the transformer and the baseline"
    ids = pred[MAIN]["id"].tolist()
    for n, d in pred.items():
        assert d["id"].tolist() == ids, f"{n}: different test rows"

    con = duckdb.connect()
    con.register("ids_df", pd.DataFrame({"id": ids}))
    path = (ROOT / "data" / "processed" / "test.csv").as_posix()
    text = con.execute(f"""
        SELECT t.id, t.body, t.body_norm, TRY_CAST(t.rate AS DOUBLE) AS rate
        FROM read_csv('{path}', header=true, all_varchar=true) t JOIN ids_df USING (id)
    """).df().drop_duplicates("id")
    df = pred[MAIN][["id", "rec_true", "band_true"]].merge(text, on="id", how="left")
    assert df["body_norm"].notna().all(), "some test ids were not found in test.csv"
    df["rec_true_name"] = df["rec_true"].map(lambda i: REC[i] if i >= 0 else None)

    for n, d in pred.items():
        P = d[[f"p_rec_{l}" for l in REC]].to_numpy()
        Q = d[[f"p_band_{l}" for l in BAND]].to_numpy()
        df[f"rec_pred_{n}"], df[f"rec_conf_{n}"] = P.argmax(1), P.max(1)
        df[f"band_pred_{n}"], df[f"band_conf_{n}"] = Q.argmax(1), Q.max(1)
    fl = pd.DataFrame([flags_for(r) for _, r in df.iterrows()])
    df = pd.concat([df.reset_index(drop=True), fl], axis=1)
    df["duplicate_text_in_sample"] = df.duplicated("body_norm", keep=False)
    flag_cols = list(fl.columns) + ["duplicate_text_in_sample"]

    rec_ok = df["rec_true"] >= 0
    band_ok = df["band_true"] >= 0
    df["rec_wrong"] = rec_ok & (df[f"rec_pred_{MAIN}"] != df["rec_true"])
    df["band_wrong"] = band_ok & (df[f"band_pred_{MAIN}"] != df["band_true"])
    df["rec_base_wrong"] = rec_ok & (df[f"rec_pred_{BASE}"] != df["rec_true"])
    df["band_far"] = band_ok & ((df[f"band_pred_{MAIN}"] - df["band_true"]).abs() == 2)
    df["rec_disagree"] = rec_ok & (df[f"rec_pred_{MAIN}"] != df[f"rec_pred_{BASE}"])
    df["rec_high_conf_wrong"] = df["rec_wrong"] & (df[f"rec_conf_{MAIN}"] >= 0.9)

    # ---- error rate per flag ----
    rows = [{"flag": "ALL rows", "n_rows": len(df),
             "rec_error_rate": df.loc[rec_ok, "rec_wrong"].mean(), "band_error_rate": df.loc[band_ok, "band_wrong"].mean(),
             "rec_error_rate_baseline": df.loc[rec_ok, "rec_base_wrong"].mean()}]
    for c in flag_cols:
        m = df[c]
        rows.append({"flag": c, "n_rows": int(m.sum()),
                     "rec_error_rate": df.loc[m & rec_ok, "rec_wrong"].mean() if (m & rec_ok).any() else np.nan,
                     "band_error_rate": df.loc[m & band_ok, "band_wrong"].mean() if (m & band_ok).any() else np.nan,
                     "rec_error_rate_baseline": df.loc[m & rec_ok, "rec_base_wrong"].mean() if (m & rec_ok).any() else np.nan})
    rates = pd.DataFrame(rows)
    rates.to_csv(OUT / "error_rates_by_flag.csv", index=False, encoding="utf-8-sig")
    print(rates.round(4).to_string(index=False))

    # ---- stratified sample of mistakes to read ----
    strata = [  # (stratum name, head, boolean mask over df, how many)
        ("rec: high-confidence error (conf>=0.9)", "rec", df["rec_high_conf_wrong"], 8),
        ("rec: transformer wrong, baseline right", "rec", df["rec_wrong"] & ~df["rec_base_wrong"], 8),
        ("rec: baseline wrong, transformer right", "rec", ~df["rec_wrong"] & df["rec_base_wrong"] & rec_ok, 6),
        ("rec: error on short text (<=3 words)", "rec", df["rec_wrong"] & df["short_le_3_words"], 5),
        ("rec: error with negation", "rec", df["rec_wrong"] & df["negation"], 5),
        ("rec: error with contrast word (mixed opinion)", "rec", df["rec_wrong"] & df["contrast_mixed"], 5),
        ("rec: possible contradictory label", "rec", df["rec_wrong"] & df["possible_contradiction_label"], 6),
        ("rec: spelling variation", "rec", df["rec_wrong"] & (df["elongated_letters"] | df["arabic_letter_variants"]), 4),
        ("rec: duplicated text in the sample", "rec", df["rec_wrong"] & df["duplicate_text_in_sample"], 3),
        ("band: error near a band boundary (rate within 0.25 of 2.5 or 4.0)", "band", df["band_wrong"] & df["band_boundary_rate"], 8),
        ("band: distance-2 error (low vs high)", "band", df["band_far"], 6),
        ("band: adjacent error with confidence>=0.9", "band", df["band_wrong"] & ~df["band_far"] & (df[f"band_conf_{MAIN}"] >= 0.9), 4),
        ("band: error on short text (<=3 words)", "band", df["band_wrong"] & df["short_le_3_words"], 3),
    ]
    taken, parts = set(), []
    for name, head, mask, k in strata:
        pool = df[mask & ~df["id"].isin(taken)]
        pick = pool.sample(n=min(k, len(pool)), random_state=SEED)
        taken |= set(pick["id"])
        parts.append(pd.DataFrame({
            "stratum": name, "head": head, "id": pick["id"], "text": pick["body"],
            "true": [(REC if head == "rec" else BAND)[i] if i >= 0 else "" for i in pick[f"{head}_true"]],
            "transformer_pred": [(REC if head == "rec" else BAND)[i] for i in pick[f"{head}_pred_{MAIN}"]],
            "transformer_conf": pick[f"{head}_conf_{MAIN}"].round(3),
            "baseline_pred": [(REC if head == "rec" else BAND)[i] for i in pick[f"{head}_pred_{BASE}"]],
            "baseline_conf": pick[f"{head}_conf_{BASE}"].round(3),
            "rate": pick["rate"], "n_words": pick["body_norm"].str.split().str.len(),
            "your_category": "", "your_note": ""}))
        print(f"{name}: eligible {int(mask.sum())}, sampled {len(pick)}")
    sample = pd.concat(parts, ignore_index=True)
    sample.to_csv(OUT / "review_sample.csv", index=False, encoding="utf-8-sig")
    print("\nrows to read:", len(sample), "->", OUT / "review_sample.csv")


if __name__ == "__main__":
    main()
