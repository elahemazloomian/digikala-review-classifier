"""Tests for the Digikala review API (app/main.py).

Run from the project root:   python -m pytest tests -v
The client is used as a context manager, so the models are loaded once (about 10 s) for the whole file.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so that "import app.main" works from any folder

from app import main  # noqa: E402

GOOD_REVIEW = "از خریدم خیلی راضی‌ام، سریع رسید و حتماً پیشنهاد می‌کنم"


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:  # "with" runs the startup code that loads the models
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_after_startup(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_normalize_arabic_letters_and_digits(client):
    # Arabic kaf (U+0643) -> Persian kaf (U+06A9); Persian digit one (U+06F1) -> "1"
    r = client.post("/normalize", json={"text": "\u0643\u06F1"})
    assert r.status_code == 200
    assert r.json()["normalized"] == "\u06A91"


def test_empty_text_is_422(client):
    r = client.post("/predict/tfidf", json={"text": ""})
    assert r.status_code == 422


def test_whitespace_only_text_is_422(client):
    r = client.post("/predict/tfidf", json={"text": "   "})
    assert r.status_code == 422


def test_too_long_text_is_422(client):
    r = client.post("/predict/tfidf", json={"text": "a" * (main.MAX_TEXT_CHARS + 1)})
    assert r.status_code == 422


def test_extra_field_like_a_label_is_422(client):
    r = client.post("/predict/tfidf", json={"text": GOOD_REVIEW, "label": "recommended"})
    assert r.status_code == 422


def test_unknown_model_is_400(client):
    r = client.post("/predict", json={"text": GOOD_REVIEW, "model": "no_such_model"})
    assert r.status_code == 400


def test_unknown_tokenizer_is_400(client):
    r = client.post("/tokenize", json={"text": GOOD_REVIEW, "tokenizer": "no_such_tokenizer"})
    assert r.status_code == 400


def test_predict_tfidf_shape(client):
    r = client.post("/predict/tfidf", json={"text": GOOD_REVIEW})
    assert r.status_code == 200
    body = r.json()
    assert body["recommendation"]["label"] in main.pr.REC
    assert body["rating_band"]["label"] in main.pr.BAND
    assert abs(sum(body["recommendation"]["probs"].values()) - 1.0) < 1e-4


def test_predict_transformer_shape(client):
    r = client.post("/predict/transformer", json={"text": GOOD_REVIEW})
    assert r.status_code == 200
    body = r.json()
    assert body["recommendation"]["label"] in main.pr.REC
    assert body["rating_band"]["label"] in main.pr.BAND
    assert body["token_count"] > 0


def test_missing_model_gives_503(client, monkeypatch):
    # pretend the Transformer failed to load at startup
    monkeypatch.setitem(main.STATE, "transformer", None)
    r = client.post("/predict/transformer", json={"text": GOOD_REVIEW})
    assert r.status_code == 503
    assert client.get("/ready").status_code == 503
