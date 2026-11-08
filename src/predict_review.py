"""Offline prediction for ONE review: TF-IDF model and fine-tuned Transformer, with explanations.

Run from the project root, for example:
    python src/predict_review.py --text "کیفیت محصول خیلی خوب بود اما بسته‌بندی ضعیف بود"
    python src/predict_review.py --demo
    python src/predict_review.py --file my_reviews.txt --json results.json      (one review per line)
Needs only saved files: artifacts/tfidf_word_*.joblib and artifacts/<run>/ (best.pt, tokenizer, config.json).
The review text is the only input; no rating or label is ever given to a model.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use saved files only; no internet needed once config.json is saved

import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from text_normalize import normalize_text

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REC = ["no_idea", "not_recommended", "recommended"]
BAND = ["low", "medium", "high"]
MAX_OCCLUDE_WORDS = 40

DEMO = [
    ("short", "خوبه"),
    ("ambiguous", "بد نیست"),
    ("ambiguous / neutral", "متناسب با قیمتش هست"),
    ("negative", "اصلا نخرید، کیفیتش افتضاحه و پولتون رو هدر نده"),
    ("mixed opinion", "کیفیت محصول خیلی خوب بود اما بسته‌بندی ضعیف بود"),
    ("negation", "جنس خوبی نداره و خیلی زود پاره شد"),
    ("positive", "از خریدم خیلی راضی‌ام، سریع رسید و دقیقاً مثل عکس بود. حتماً پیشنهاد می‌کنم"),
    ("spelling variation", "کالا کاملان بدونه کیفیت"),
]


class TwoHeadModel(nn.Module):
    """Same structure as ReviewClassifier in finetune_transformer.py, built from a saved config (no weight download)."""

    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        h = config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.rec_head = nn.Linear(h, len(REC))
        self.band_head = nn.Linear(h, len(BAND))

    def forward(self, enc):
        h = self.dropout(self.encoder(**enc).last_hidden_state[:, 0])
        return self.rec_head(h), self.band_head(h)


def load_transformer(run_dir):
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    try:
        config = AutoConfig.from_pretrained(run_dir)
    except Exception:
        config = AutoConfig.from_pretrained(ckpt["model_name"])  # first time only (cached copy or internet)
        config.save_pretrained(run_dir)
        print(f"[info] saved config.json into {run_dir}", file=sys.stderr)
    model = TwoHeadModel(config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, AutoTokenizer.from_pretrained(run_dir), ckpt["max_length"]


def transformer_probs(model, tokenizer, texts, max_length):
    out_r, out_b = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            lr_, lb_ = model(enc)
            out_r.append(torch.softmax(lr_, 1).numpy())
            out_b.append(torch.softmax(lb_, 1).numpy())
    return np.concatenate(out_r), np.concatenate(out_b)


def occlusion(model, tokenizer, max_length, text, pred_rec, pred_band, k=3):
    """Leave-one-word-out: how much does the probability of the predicted class drop when a word is removed?"""
    words = text.split()[:MAX_OCCLUDE_WORDS]
    if len(words) < 2:
        return None
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    full_r, full_b = transformer_probs(model, tokenizer, [" ".join(words)], max_length)
    pr, pb = transformer_probs(model, tokenizer, variants, max_length)
    res = {}
    for name, full, p, c in (("recommendation", full_r[0], pr, pred_rec), ("rating_band", full_b[0], pb, pred_band)):
        imp = full[c] - p[:, c]
        order = np.argsort(imp)
        res[name] = {"supports": [(words[i], float(imp[i])) for i in order[::-1][:k] if imp[i] > 0],
                     "opposes": [(words[i], float(imp[i])) for i in order[:k] if imp[i] < 0]}
    return res


def tfidf_part(vecs, text, k=5):
    out = {}
    for head, labels in (("recommendation", REC), ("rating_band", BAND)):
        vec, clf = vecs[head]
        x = vec.transform([text])
        proba = clf.predict_proba(x)[0]
        cols = [list(clf.classes_).index(l) for l in labels]
        proba = proba[cols]
        c = int(proba.argmax())
        coo = x.tocoo()
        names = vec.get_feature_names_out()
        contrib = coo.data * clf.coef_[cols[c], coo.col]
        order = np.argsort(contrib)
        out[head] = {"label": labels[c], "probs": {l: float(p) for l, p in zip(labels, proba)},
                     "supports": [(names[coo.col[i]], float(contrib[i])) for i in order[::-1][:k] if contrib[i] > 0],
                     "opposes": [(names[coo.col[i]], float(contrib[i])) for i in order[:k] if contrib[i] < 0]}
    return out


def fmt_probs(d):
    return ", ".join(f"{k} {v:.2f}" for k, v in d.items())


def fmt_words(pairs):
    return ", ".join(f"{w} ({v:+.2f})" for w, v in pairs) if pairs else "-"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--file", help="text file, one review per line (UTF-8)")
    g.add_argument("--demo", action="store_true", help="run the built-in short / ambiguous / negative / mixed examples")
    ap.add_argument("--run-dir", default="artifacts/ft_200k", help="folder with best.pt of the Transformer")
    ap.add_argument("--no-transformer", action="store_true")
    ap.add_argument("--json", default=None, help="also save all results to this UTF-8 json file")
    args = ap.parse_args()

    if args.demo:
        items = DEMO
    elif args.file:
        items = [("file", ln.strip()) for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        items = [("input", args.text)]

    vecs = {h: (joblib.load(ARTIFACTS / f"tfidf_word_{f}_vectorizer.joblib"), joblib.load(ARTIFACTS / f"tfidf_word_{f}_logreg.joblib"))
            for h, f in (("recommendation", "recommendation_status"), ("rating_band", "rating_band"))}
    tr = None if args.no_transformer else load_transformer(ROOT / args.run_dir)
    if tr:
        transformer_probs(tr[0], tr[1], ["warm up"], tr[2])  # first call is slow; do it before timing

    results = []
    for kind, raw in items:
        norm = normalize_text(raw)
        if not norm:
            print(f"\n(empty review skipped: {raw!r})")
            continue
        r = {"kind": kind, "text": raw, "normalized": norm}
        t0 = time.time()
        r["tfidf"] = tfidf_part(vecs, norm)
        r["tfidf"]["latency_ms"] = round((time.time() - t0) * 1000, 1)
        if tr:
            model, tok, max_len = tr
            t0 = time.time()
            n_tok = len(tok(norm, truncation=False)["input_ids"])
            pr, pb = transformer_probs(model, tok, [norm], max_len)
            c_r, c_b = int(pr[0].argmax()), int(pb[0].argmax())
            r["transformer"] = {"tokens": n_tok, "truncated": n_tok > max_len, "max_length": max_len,
                                "recommendation": {"label": REC[c_r], "probs": {l: float(p) for l, p in zip(REC, pr[0])}},
                                "rating_band": {"label": BAND[c_b], "probs": {l: float(p) for l, p in zip(BAND, pb[0])}},
                                "latency_ms": round((time.time() - t0) * 1000, 1)}
            r["transformer"]["explanation"] = occlusion(model, tok, max_len, norm, c_r, c_b)
        results.append(r)

        print("\n" + "=" * 78)
        print(f"[{kind}] {raw}")
        if norm != raw:
            print(f"normalized: {norm}")
        for name, part in (("TF-IDF + logistic regression", r["tfidf"]), ("Fine-tuned Transformer", r.get("transformer"))):
            if part is None:
                continue
            print(f"\n--- {name} ---")
            if name.startswith("Fine"):
                print(f"tokens: {part['tokens']} (max {part['max_length']}, truncated: {'yes' if part['truncated'] else 'no'})"
                      f" | latency {part['latency_ms']} ms")
            for head in ("recommendation", "rating_band"):
                h = part[head]
                print(f"{head}: {h['label']}   [{fmt_probs(h['probs'])}]")
            if name.startswith("TF-IDF"):
                for head in ("recommendation", "rating_band"):
                    print(f"  {head} - terms pushing toward the prediction: {fmt_words(part[head]['supports'])}")
                    print(f"  {head} - terms pushing against it: {fmt_words(part[head]['opposes'])}")
            else:
                ex = part["explanation"]
                if ex is None:
                    print("  (explanation needs at least 2 words)")
                else:
                    for head in ("recommendation", "rating_band"):
                        print(f"  {head} - removing this word lowers the predicted probability: {fmt_words(ex[head]['supports'])}")
                        print(f"  {head} - removing this word raises it: {fmt_words(ex[head]['opposes'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nsaved:", args.json)


if __name__ == "__main__":
    main()
