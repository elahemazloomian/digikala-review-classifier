"""Evaluate the best saved two-head checkpoint on the VALIDATION file.

Run from the project root, for example:
    python src/evaluate_transformer.py --run-dir artifacts/ft_200k --rows 100000
Leave out --rows to use every validation row (slow). The test file is never read here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from finetune_transformer import (IGNORE, ROOT, ReviewClassifier, ReviewData, evaluate,
                                  load_rows, make_collate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(run_dir)
    model = ReviewClassifier(ckpt["model_name"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print("checkpoint:", run_dir / "best.pt", "| max_length:", ckpt["max_length"], flush=True)

    texts, rec, band = load_rows("validation.csv", args.rows, args.seed)
    order = np.argsort([len(t) for t in texts], kind="stable")  # similar lengths together: less padding
    texts, rec, band = [texts[i] for i in order], rec[order], band[order]
    print("validation rows:", len(texts), flush=True)
    loader = DataLoader(ReviewData(texts, rec, band), batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_collate(tokenizer, ckpt["max_length"]))

    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)
    res = evaluate(model, loader, device, use_amp, loss_fn, loss_fn)

    for name, labels in (("rec", ckpt["rec_labels"]), ("band", ckpt["band_labels"])):
        m = res[name]
        print(f"\n== {name} ({m['n']:,} rows) ==")
        print(f"accuracy {m['accuracy']:.4f} | balanced acc {m['balanced_accuracy']:.4f} | macro P {m['macro_precision']:.4f} "
              f"| macro R {m['macro_recall']:.4f} | macro F1 {m['macro_f1']:.4f} | weighted F1 {m['weighted_f1']:.4f}")
        print(f"MCC {m['mcc']:.4f} | NLL {m['nll']:.4f} | ECE {m['ece']:.4f}")
        if "ordinal_distance_error" in m:
            print(f"ordinal distance error {m['ordinal_distance_error']:.4f} | distance-2 share "
                  f"{m['share_distance_2_errors']:.4f} | QWK {m['quadratic_weighted_kappa']:.4f}")
        for lab in labels:
            c = m["per_class"][lab]
            print(f"  {lab:16s} P {c['precision']:.3f}  R {c['recall']:.3f}  F1 {c['f1']:.3f}  n {c['support']:,}")
        print("  confusion matrix (rows = true, columns = predicted):", labels)
        for lab, row in zip(labels, m["confusion_matrix"]):
            print(f"  {lab:16s}", row)

    out = ROOT / "reports" / f"{run_dir.name}_eval_validation.json"
    res.update({"checkpoint": str(run_dir / "best.pt"), "rows_requested": args.rows})
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\nsaved:", out)


if __name__ == "__main__":
    main()