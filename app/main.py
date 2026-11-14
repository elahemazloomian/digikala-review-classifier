"""Digikala review API (optional Phase 2).

Serves the saved TF-IDF and fine-tuned Transformer models over HTTP.
It reuses the tested code of src/predict_review.py, so the API and the offline script
give the same predictions. Models are loaded ONCE at startup, never per request.
The only input is the review text: no rating or label is ever accepted.
Raw reviews are never written to the log (only lengths and timings).
"""
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import predict_review as pr
from text_normalize import normalize_text

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
RUN_NAME = os.environ.get("TRANSFORMER_RUN", "ft_200k")
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "5000"))

log = logging.getLogger("digikala-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

STATE = {"tfidf": None, "transformer": None, "load_seconds": None}


@asynccontextmanager
async def lifespan(app):
    t0 = time.time()
    try:
        STATE["tfidf"] = {
            h: (joblib.load(ARTIFACTS / f"tfidf_word_{f}_vectorizer.joblib"),
                joblib.load(ARTIFACTS / f"tfidf_word_{f}_logreg.joblib"))
            for h, f in (("recommendation", "recommendation_status"), ("rating_band", "rating_band"))}
        tfidf_probs(STATE["tfidf"], "warm up")  # first call is slow; do it before timing
        log.info("tfidf models loaded")
    except Exception as e:  # missing or broken files -> service stays up but not ready (503)
        log.error("tfidf load failed: %s", type(e).__name__)
    try:
        STATE["transformer"] = pr.load_transformer(ARTIFACTS / RUN_NAME)
        model, tok, max_len = STATE["transformer"]
        pr.transformer_probs(model, tok, ["warm up"], max_len)  # first call is slow
        log.info("transformer loaded: %s", RUN_NAME)
    except Exception as e:
        STATE["transformer"] = None
        log.error("transformer load failed: %s", type(e).__name__)
    STATE["load_seconds"] = round(time.time() - t0, 1)
    yield


app = FastAPI(title="Digikala review classifier", version="1.0.0", lifespan=lifespan)


# ---------- request / response shapes ----------
class TextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # labels or ratings in the body are rejected (422)
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)


class PredictRequest(TextRequest):
    model: str = "transformer"  # transformer | tfidf | compare


class TokenizeRequest(TextRequest):
    tokenizer: str = "transformer"


class Head(BaseModel):
    label: str
    probs: dict[str, float]


class PredictResponse(BaseModel):
    model: str
    model_version: str
    recommendation: Head
    rating_band: Head
    token_count: int
    latency_ms: float


# ---------- helpers ----------
def prepare(text: str) -> str:
    norm = normalize_text(text)
    if not norm:
        raise HTTPException(422, "text is empty after normalization")
    return norm


def need(name: str):
    if STATE[name] is None:
        raise HTTPException(503, f"{name} model is not ready")
    return STATE[name]


def tfidf_probs(vecs, norm: str) -> dict:
    """Same probabilities as pr.tfidf_part, without the word-contribution lists (too slow for an API)."""
    out = {}
    for head, labels in (("recommendation", pr.REC), ("rating_band", pr.BAND)):
        vec, clf = vecs[head]
        proba = clf.predict_proba(vec.transform([norm]))[0]
        proba = proba[[list(clf.classes_).index(l) for l in labels]]
        c = int(proba.argmax())
        out[head] = {"label": labels[c], "probs": {l: float(p) for l, p in zip(labels, proba)}}
    return out


def run_tfidf(norm: str) -> dict:
    vecs = need("tfidf")
    t0 = time.time()
    part = tfidf_probs(vecs, norm)
    return {"model": "tfidf_logreg", "model_version": "tfidf_word_logreg",
            "recommendation": {"label": part["recommendation"]["label"], "probs": part["recommendation"]["probs"]},
            "rating_band": {"label": part["rating_band"]["label"], "probs": part["rating_band"]["probs"]},
            "token_count": len(norm.split()),  # for TF-IDF: number of words
            "latency_ms": round((time.time() - t0) * 1000, 1)}


def run_transformer(norm: str) -> dict:
    model, tok, max_len = need("transformer")
    t0 = time.time()
    n_tok = len(tok(norm, truncation=False)["input_ids"])
    p_r, p_b = pr.transformer_probs(model, tok, [norm], max_len)
    c_r, c_b = int(p_r[0].argmax()), int(p_b[0].argmax())
    return {"model": "transformer_finetuned", "model_version": RUN_NAME,
            "recommendation": {"label": pr.REC[c_r], "probs": {l: float(p) for l, p in zip(pr.REC, p_r[0])}},
            "rating_band": {"label": pr.BAND[c_b], "probs": {l: float(p) for l, p in zip(pr.BAND, p_b[0])}},
            "token_count": n_tok,
            "latency_ms": round((time.time() - t0) * 1000, 1)}


def run_compare(norm: str) -> dict:
    a, b = run_tfidf(norm), run_transformer(norm)
    return {"tfidf": a, "transformer": b,
            "agree": {"recommendation": a["recommendation"]["label"] == b["recommendation"]["label"],
                      "rating_band": a["rating_band"]["label"] == b["rating_band"]["label"]}}


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    parts = {"tfidf": STATE["tfidf"] is not None, "transformer": STATE["transformer"] is not None}
    body = {"ready": all(parts.values()), "models": parts, "load_seconds": STATE["load_seconds"]}
    return JSONResponse(body, status_code=200 if body["ready"] else 503)


@app.post("/normalize")
def normalize(req: TextRequest):
    return {"original": req.text, "normalized": normalize_text(req.text)}


@app.post("/tokenize")
def tokenize(req: TokenizeRequest):
    if req.tokenizer != "transformer":
        raise HTTPException(400, f"unknown tokenizer: {req.tokenizer!r} (available: transformer)")
    _, tok, max_len = need("transformer")
    norm = prepare(req.text)
    full = len(tok(norm, truncation=False)["input_ids"])
    enc = tok(norm, truncation=True, max_length=max_len)
    return {"normalized": norm, "tokens": tok.convert_ids_to_tokens(enc["input_ids"]),
            "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "token_count": full, "max_length": max_len, "truncated": full > max_len}


@app.post("/predict/tfidf", response_model=PredictResponse)
def predict_tfidf(req: TextRequest):
    return run_tfidf(prepare(req.text))


@app.post("/predict/transformer", response_model=PredictResponse)
def predict_transformer(req: TextRequest):
    return run_transformer(prepare(req.text))


@app.post("/predict/compare")
def predict_compare(req: TextRequest):
    return run_compare(prepare(req.text))


@app.post("/predict")
def predict(req: PredictRequest):
    norm = prepare(req.text)
    if req.model == "transformer":
        return run_transformer(norm)
    if req.model == "tfidf":
        return run_tfidf(norm)
    if req.model == "compare":
        return run_compare(norm)
    raise HTTPException(400, f"unknown model: {req.model!r} (available: transformer, tfidf, compare)")


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    log.error("unexpected error on %s: %s", request.url.path, type(exc).__name__)  # never log the review text
    return JSONResponse({"detail": "internal server error"}, status_code=500)
