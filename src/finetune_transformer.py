"""Fine-tune one pretrained encoder with TWO heads (recommendation, rating band).

Run from the project root, for example a small test:
    python src/finetune_transformer.py --train-rows 20000 --val-rows 5000 --epochs 1
Continue after an interruption (same options + --resume):
    python src/finetune_transformer.py --train-rows 20000 --val-rows 5000 --epochs 1 --resume
The test file is never read here.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REC_LABELS = ["no_idea", "not_recommended", "recommended"]
BAND_LABELS = ["low", "medium", "high"]
REC_ID = {name: i for i, name in enumerate(REC_LABELS)}
BAND_ID = {name: i for i, name in enumerate(BAND_LABELS)}
IGNORE = -100  # label value meaning "this target is missing for this row"

# options that must be identical when resuming (otherwise the saved state does not fit)
MUST_MATCH = ["model_name", "max_length", "train_rows", "batch_size", "grad_accum", "lr_encoder",
              "lr_head", "weight_decay", "warmup_ratio", "unfreeze_last", "freeze_embeddings",
              "epochs", "seed", "bucket_chunk"]


# ----------------------------------------------------------------- data
def load_rows(file_name, max_rows, seed):
    path = (PROCESSED / file_name).as_posix()
    sample = f"USING SAMPLE reservoir({max_rows} ROWS) REPEATABLE ({seed})" if max_rows else ""
    df = duckdb.connect().execute(f"""
        SELECT body_norm, recommendation_status AS rec, rating_band AS band
        FROM read_csv('{path}', header=true, all_varchar=true)
        WHERE body_norm IS NOT NULL
          AND (recommendation_status IS NOT NULL OR rating_band IS NOT NULL)
        {sample}
    """).df()
    rec = df["rec"].map(REC_ID).fillna(IGNORE).astype(int).to_numpy()
    band = df["band"].map(BAND_ID).fillna(IGNORE).astype(int).to_numpy()
    return df["body_norm"].tolist(), rec, band


class ReviewData(torch.utils.data.Dataset):
    def __init__(self, texts, rec, band):
        self.texts, self.rec, self.band = texts, rec, band

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], int(self.rec[i]), int(self.band[i])


def make_collate(tokenizer, max_length):
    def collate(batch):
        texts, rec, band = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt")  # pad only to the longest in the batch
        return enc, torch.tensor(rec), torch.tensor(band)
    return collate


def epoch_batches(lengths, batch_size, seed, epoch, chunk_batches):
    """Batches of one epoch. Depends only on (lengths, seed, epoch) so a resumed run sees the same order.

    chunk_batches > 0: shuffle everything, cut into big chunks, sort each chunk by text length, cut
    into batches, then shuffle the batches. Texts in a batch have similar length, so little padding.
    """
    n = len(lengths)
    rng = np.random.default_rng(seed + epoch)
    perm = rng.permutation(n)
    if chunk_batches <= 0:
        return [perm[i:i + batch_size].tolist() for i in range(0, n, batch_size)]
    batches, size = [], batch_size * chunk_batches
    for i in range(0, n, size):
        chunk = perm[i:i + size]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        batches += [chunk[k:k + batch_size].tolist() for k in range(0, len(chunk), batch_size)]
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def class_weights(labels, n_classes):
    counts = np.bincount(labels[labels != IGNORE], minlength=n_classes).astype(float)
    return torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float)


# ---------------------------------------------------------------- model
class ReviewClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.rec_head = nn.Linear(hidden, len(REC_LABELS))
        self.band_head = nn.Linear(hidden, len(BAND_LABELS))

    def forward(self, enc):
        h = self.encoder(**enc).last_hidden_state[:, 0]  # vector of the first ([CLS]) token
        h = self.dropout(h)
        return self.rec_head(h), self.band_head(h)


def masked_loss(logits, labels, loss_fn):
    if (labels != IGNORE).any():
        return loss_fn(logits, labels)
    return logits.sum() * 0.0  # no valid target for this head in this batch


def apply_freezing(model, n_last, freeze_embeddings):
    if n_last >= 0:  # train only the last n_last layers (and the heads)
        for p in model.encoder.parameters():
            p.requires_grad = False
        layers = model.encoder.encoder.layer
        for layer in layers[len(layers) - n_last:]:
            for p in layer.parameters():
                p.requires_grad = True
    if freeze_embeddings:  # the big word-embedding table: ~half of all parameters
        model.encoder.embeddings.word_embeddings.weight.requires_grad = False


# ------------------------------------------------------------ evaluation
def expected_calibration_error(probs, y_true, n_bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def head_metrics(y_true, y_pred, probs, labels, ordinal=False):
    k = len(labels)
    idx = list(range(k))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=idx, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "nll": float(log_loss(y_true, probs, labels=idx)),
        "ece": expected_calibration_error(probs, y_true),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                            "support": int(s[i])} for i, lab in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }
    if ordinal:
        dist = np.abs(y_true - y_pred)
        out["ordinal_distance_error"] = float(dist.mean())
        out["share_distance_2_errors"] = float((dist == 2).mean())
        out["quadratic_weighted_kappa"] = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    return out


@torch.no_grad()
def evaluate(model, loader, device, use_amp, rec_loss_fn, band_loss_fn):
    model.eval()
    store = {"rec": ([], [], []), "band": ([], [], [])}  # true, predicted, probabilities
    losses = []
    for enc, rec, band in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        rec, band = rec.to(device), band.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            lr_, lb_ = model(enc)
        lr_, lb_ = lr_.float(), lb_.float()
        losses.append((masked_loss(lr_, rec, rec_loss_fn) + masked_loss(lb_, band, band_loss_fn)).item())
        for name, logits, y in (("rec", lr_, rec), ("band", lb_, band)):
            m = y != IGNORE
            if m.any():
                store[name][0].append(y[m].cpu().numpy())
                store[name][1].append(logits.argmax(1)[m].cpu().numpy())
                store[name][2].append(torch.softmax(logits[m], dim=1).cpu().numpy())
    model.train()
    out = {"val_loss": float(np.mean(losses))}
    for name, labels, ordinal in (("rec", REC_LABELS, False), ("band", BAND_LABELS, True)):
        t, p, pr = (np.concatenate(x) for x in store[name])
        out[name] = head_metrics(t, p, pr, labels, ordinal)
    out["mean_macro_f1"] = (out["rec"]["macro_f1"] + out["band"]["macro_f1"]) / 2
    return out


def save_state(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # the old file is replaced only after the new one is complete


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="HooshvareLab/bert-fa-base-uncased")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--train-rows", type=int, default=None)
    ap.add_argument("--val-rows", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--unfreeze-last", type=int, default=-1, help="-1 = train every layer")
    ap.add_argument("--freeze-embeddings", action="store_true")
    ap.add_argument("--bucket-chunk", type=int, default=50, help="0 = plain random batches")
    ap.add_argument("--eval-every-steps", type=int, default=0, help="0 = only at the end of each epoch")
    ap.add_argument("--rest-minutes", type=float, default=0.0, help="pause after each checkpoint")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after-steps", type=int, default=0, help="testing only: simulate an interruption")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default="artifacts/transformer_run")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    hist_path = ROOT / "reports" / f"{Path(args.out_dir).name}_history.json"
    hist_path.parent.mkdir(exist_ok=True)
    print("device:", device, "| mixed precision:", use_amp, flush=True)

    tr_texts, tr_rec, tr_band = load_rows("train_supervised.csv", args.train_rows, args.seed)
    va_texts, va_rec, va_band = load_rows("validation.csv", args.val_rows, args.seed)
    print("train rows:", len(tr_texts), "| validation rows:", len(va_texts), flush=True)
    train_lengths = np.array([len(t) for t in tr_texts])
    order = np.argsort([len(t) for t in va_texts], kind="stable")  # similar lengths together: less padding
    va_texts, va_rec, va_band = [va_texts[i] for i in order], va_rec[order], va_band[order]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate = make_collate(tokenizer, args.max_length)
    train_ds = ReviewData(tr_texts, tr_rec, tr_band)
    val_loader = DataLoader(ReviewData(va_texts, va_rec, va_band), batch_size=args.batch_size * 2,
                            shuffle=False, collate_fn=collate)

    w_rec, w_band = class_weights(tr_rec, len(REC_LABELS)), class_weights(tr_band, len(BAND_LABELS))
    rec_loss_fn = nn.CrossEntropyLoss(weight=w_rec.to(device), ignore_index=IGNORE)
    band_loss_fn = nn.CrossEntropyLoss(weight=w_band.to(device), ignore_index=IGNORE)

    model = ReviewClassifier(args.model_name).to(device)
    apply_freezing(model, args.unfreeze_last, args.freeze_embeddings)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: total {total:,} | trainable {trainable:,}", flush=True)

    no_decay = ("bias", "LayerNorm.weight")
    enc_params = [(n, p) for n, p in model.encoder.named_parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups = [
        {"params": [p for n, p in enc_params if not any(k in n for k in no_decay)],
         "lr": args.lr_encoder, "weight_decay": args.weight_decay},
        {"params": [p for n, p in enc_params if any(k in n for k in no_decay)],
         "lr": args.lr_encoder, "weight_decay": 0.0},
        {"params": head_params, "lr": args.lr_head, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]])
    n_batches = -(-len(train_ds) // args.batch_size)
    steps_total = max(1, (n_batches // args.grad_accum) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * steps_total), steps_total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"args": vars(args), "total_params": total, "trainable_params": trainable,
               "train_loss": [], "evals": [], "minutes": 0.0}
    best, bad_evals, step = -1.0, 0, 0
    start_epoch, start_batch = 0, 0

    if args.resume:
        if not last_path.exists():
            sys.exit(f"--resume was given but {last_path} does not exist")
        st = torch.load(last_path, map_location=device, weights_only=False)
        for k in MUST_MATCH:
            if st["args"].get(k) != vars(args).get(k):
                sys.exit(f"cannot resume: option {k} differs (saved {st['args'].get(k)!r}, now {vars(args).get(k)!r})")
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"]); scaler.load_state_dict(st["scaler"])
        history, best, bad_evals, step = st["history"], st["best"], st["bad_evals"], st["step"]
        start_epoch, start_batch = st["epoch"], st["next_batch"]
        print(f"resumed: epoch {start_epoch + 1}, batch {start_batch}, step {step}, best so far {best:.4f}", flush=True)
        del st  # free the loaded copy
    t_before = history.get("minutes", 0.0) * 60
    t_start = time.time()
    seen = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    stop = False

    def checkpoint_and_eval(epoch_next, batch_next, is_last):
        nonlocal best, bad_evals, stop
        res = evaluate(model, val_loader, device, use_amp, rec_loss_fn, band_loss_fn)
        res.update({"epoch": epoch_next, "step": step})
        history["evals"].append(res)
        r, b = res["rec"], res["band"]
        print(f"== step {step} validation | rec: acc {r['accuracy']:.4f} macroF1 {r['macro_f1']:.4f} "
              f"balAcc {r['balanced_accuracy']:.4f} ECE {r['ece']:.3f} | band: acc {b['accuracy']:.4f} "
              f"macroF1 {b['macro_f1']:.4f} balAcc {b['balanced_accuracy']:.4f} "
              f"ordErr {b['ordinal_distance_error']:.4f} QWK {b['quadratic_weighted_kappa']:.4f} "
              f"| val loss {res['val_loss']:.4f}", flush=True)
        if res["mean_macro_f1"] > best:
            best, bad_evals = res["mean_macro_f1"], 0
            torch.save({"state_dict": model.state_dict(), "model_name": args.model_name,
                        "max_length": args.max_length, "rec_labels": REC_LABELS,
                        "band_labels": BAND_LABELS}, best_path)
            tokenizer.save_pretrained(out_dir)
            print("   saved new best checkpoint", flush=True)
        else:
            bad_evals += 1
            if bad_evals >= args.patience:
                print("early stopping: no improvement for", args.patience, "evaluations", flush=True)
                stop = True
        history["minutes"] = round((t_before + time.time() - t_start) / 60, 2)
        if device.type == "cuda":
            history["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        save_state(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                               "history": history, "best": best, "bad_evals": bad_evals, "step": step,
                               "epoch": epoch_next, "next_batch": batch_next, "args": vars(args)})
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print("   checkpoint saved (last.pt)", flush=True)
        if args.rest_minutes > 0 and not stop and not is_last:
            print(f"   resting {args.rest_minutes} minutes to let the laptop cool...", flush=True)
            time.sleep(args.rest_minutes * 60)

    for epoch in range(start_epoch, args.epochs):
        batches = epoch_batches(train_lengths, args.batch_size, args.seed, epoch, args.bucket_chunk)
        skip = start_batch if epoch == start_epoch else 0
        loader = DataLoader(train_ds, batch_sampler=batches[skip:], collate_fn=collate)
        run_loss, run_n = 0.0, 0
        for j, (enc, rec, band) in enumerate(loader, start=skip):
            enc = {k: v.to(device) for k, v in enc.items()}
            rec, band = rec.to(device), band.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                lr_, lb_ = model(enc)
            loss = masked_loss(lr_.float(), rec, rec_loss_fn) + masked_loss(lb_.float(), band, band_loss_fn)
            scaler.scale(loss / args.grad_accum).backward()
            run_loss += loss.item(); run_n += 1; seen += len(rec)
            if (j + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    mem = f" | peak GPU {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB" if device.type == "cuda" else ""
                    avg = run_loss / run_n
                    print(f"epoch {epoch+1} step {step}/{steps_total} | loss {avg:.4f} | "
                          f"{seen / (time.time() - t_start):.0f} samples/s{mem}", flush=True)
                    history["train_loss"].append({"step": step, "loss": avg}); run_loss, run_n = 0.0, 0
                if args.stop_after_steps and step >= args.stop_after_steps:
                    sys.exit("simulated interruption (testing)")
                if args.eval_every_steps and step % args.eval_every_steps == 0 and (j + 1) < len(batches):
                    checkpoint_and_eval(epoch, j + 1, False)
                    if stop:
                        break
        if stop:
            break
        optimizer.zero_grad()  # drop a leftover partial accumulation at the end of the epoch
        checkpoint_and_eval(epoch + 1, 0, epoch + 1 == args.epochs)
        if stop:
            break

    print("done. total training time:", history["minutes"], "minutes", flush=True)


if __name__ == "__main__":
    main()