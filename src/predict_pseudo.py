"""Predict class probabilities for the artificial unlabeled pool with a saved two-head checkpoint.

The file train_pseudo_unlabeled.csv has NO labels, so nothing here can use them.
Run from the project root, for example:
    python src/predict_pseudo.py --run-dir artifacts/ft_200k --rest-minutes 2
Safe to stop (Ctrl+C) and run again: finished parts are skipped.
Output: artifacts/<run>/pseudo_probs/part_00000.parquet, part_00001.parquet, ...
"""
import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from finetune_transformer import PROCESSED, ROOT, ReviewClassifier, make_collate


class TextOnly(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], 0, 0  # dummy labels, only the collate function needs the shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--file", default="train_pseudo_unlabeled.csv")
    ap.add_argument("--rows", type=int, default=None, help="only the first N rows (for tests)")
    ap.add_argument("--part-rows", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--rest-minutes", type=float, default=0.0, help="pause after each finished part")
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    out_dir = run_dir / "pseudo_probs"
    out_dir.mkdir(exist_ok=True)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(run_dir)
    model = ReviewClassifier(ckpt["model_name"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    collate = make_collate(tokenizer, ckpt["max_length"])
    print("checkpoint:", run_dir / "best.pt", "| device:", device, flush=True)

    path = (PROCESSED / args.file).as_posix()
    limit = f"LIMIT {args.rows}" if args.rows else ""
    df = duckdb.connect().execute(f"""
        SELECT id, body_norm FROM read_csv('{path}', header=true, all_varchar=true)
        WHERE body_norm IS NOT NULL {limit}
    """).df()
    n_parts = -(-len(df) // args.part_rows)
    print("rows:", len(df), "| parts:", n_parts, flush=True)

    t0 = time.time()
    done_rows = 0
    for p in range(n_parts):
        out_path = out_dir / f"part_{p:05d}.parquet"
        part = df.iloc[p * args.part_rows:(p + 1) * args.part_rows].reset_index(drop=True)
        if out_path.exists():
            print(f"part {p + 1}/{n_parts}: already done, skipped", flush=True)
            continue
        order = np.argsort(part["body_norm"].str.len().to_numpy(), kind="stable")  # less padding
        texts = part["body_norm"].iloc[order].tolist()
        loader = DataLoader(TextOnly(texts), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        rec_p, band_p = [], []
        with torch.no_grad():
            for enc, _, _ in loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    lr_, lb_ = model(enc)
                rec_p.append(torch.softmax(lr_.float(), 1).cpu().numpy())
                band_p.append(torch.softmax(lb_.float(), 1).cpu().numpy())
        rec_p, band_p = np.concatenate(rec_p), np.concatenate(band_p)
        res = pd.DataFrame({"id": part["id"].iloc[order].to_numpy()})
        for i, name in enumerate(ckpt["rec_labels"]):
            res[f"p_rec_{name}"] = rec_p[:, i].astype("float32")
        for i, name in enumerate(ckpt["band_labels"]):
            res[f"p_band_{name}"] = band_p[:, i].astype("float32")
        tmp = out_path.with_suffix(".tmp")
        res.to_parquet(tmp)
        tmp.replace(out_path)
        done_rows += len(part)
        print(f"part {p + 1}/{n_parts} finished | {done_rows / (time.time() - t0):.0f} rows/s", flush=True)
        if args.rest_minutes > 0 and p < n_parts - 1:
            print(f"   resting {args.rest_minutes} minutes...", flush=True)
            time.sleep(args.rest_minutes * 60)
    print("all parts done:", out_dir, flush=True)


if __name__ == "__main__":
    main()