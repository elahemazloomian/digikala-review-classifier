# Digikala Review Classifier

Two predictions from the text of one Persian product review (Digikala):

* **`recommendation_status`**: `no_idea`, `not_recommended`, `recommended`
* **`rating_band`**, derived from the numeric `rate`: `low` (1 <= rate < 2.5), `medium` (2.5 <= rate < 4.0), `high` (4.0 <= rate <= 5.0).
  `rate = 0` means "no rating" and gets no band. `rate` is never used as a model input.

The project compares a TF-IDF + logistic regression baseline with one fine-tuned Persian BERT encoder that has two output heads,
and tests whether a pseudo-label (semi-supervised) step helps. All numbers below are read from files in `reports/`.

## 1. Data and split

The two raw files are **not** in this repository. Put them here before running the preprocessing notebook:

```
data/digikala-comments.csv
data/digikala-products.csv
```

| item | value |
|---|---|
| raw records | 6,156,289 |
| clean records (after the cleaning rules in `notebooks/00_preprocessing_and_split.ipynb`) | 6,152,673 rows, 331,597 products |
| split | product-grouped 70 / 15 / 15 by cumulative review count, products ordered by `md5(product_id|2026)` |
| train | 231,933 products, 4,306,881 rows |
| validation | 50,149 products, 922,896 rows |
| test | 49,515 products, 922,896 rows |
| artificial unlabeled pool | 736,379 labeled training rows whose recommendation label was hidden on purpose (kept in `data/interim/masked_ids.parquet` for auditing only) |

No product appears in two splits. The full record of seeds, thresholds and counts is in `reports/split_manifest.json`.

## 2. Repository map

```
notebooks/
  00_preprocessing_and_split.ipynb   cleaning, normalisation, product-grouped split, masking (extra)
  01_descriptive_analysis.ipynb      required: descriptive analysis of the full data
  02_baselines.ipynb                 required: TF-IDF baselines (reads saved results)
  03_transfer_learning_finetuning.ipynb   required: fine-tuned Transformer (reads saved results)
  04_semi_supervised.ipynb           required: pseudo-label audit and semi-supervised comparison
  05_label_noise_check.ipynb         extra: label noise check
  extra_token_length_study.ipynb     extra: token lengths used to choose max_length
  archive_baseline_sample_experiments.ipynb   archived sample experiments (word vs character n-grams)
src/
  text_normalize.py                  Persian text normalisation
  train_tfidf_baseline.py            baseline training
  finetune_transformer.py            Transformer fine-tuning (two heads)
  evaluate_transformer.py            validation evaluation of a saved checkpoint
  predict_pseudo.py                  predict the unlabeled pool
  make_pseudo_labels.py              select pseudo-labels (threshold 0.90)
  evaluate_test.py                   test evaluation, each model once
  compare_test.py                    paired bootstrap comparison
  export_errors.py                   error tables
  predict_review.py                  offline prediction for one review, with explanations
app/                                 FastAPI service (optional part)
tests/                               API tests with TestClient (optional part)
Dockerfile, .dockerignore            API image (optional part)
requirements-api.txt                 packages of the API image
reports/                             saved results (json, csv, figures); no large files
artifacts/                           trained models; NOT in git
data/                                raw and processed data; NOT in git
```

## 3. Installation

Python 3.11 is the tested version (3.11.18).

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m ipykernel install --user --name digikala --display-name "Python (digikala)"
```

Notes:

* The notebooks use the kernel name `digikala`.
* The models were **trained on Windows with `torch 2.14.1+cu132`** (CUDA build, 4 GB GPU). The notebooks and the offline prediction script were **tested on Ubuntu with the CPU build `torch 2.9.1+cpu`**. Loading the saved checkpoint worked with this version.
* The first `pip` line is the official CPU wheel source. On the machine used for testing that server was not reachable, so the CPU build was installed from a mirror instead (`https://mirrors.aliyun.com/pytorch-wheels/cpu/`). The result is the same version.
* Training on a GPU needs a CUDA build of `torch` instead.

## 4. Execution order

Run from the project root. Every script prints its options with `python src/<name>.py --help`; the exact settings used for the reported runs are stored in the `args` field of the history files in `reports/`.

| step | what | how |
|---|---|---|
| 1 | cleaning, normalisation, split, masking | `notebooks/00_preprocessing_and_split.ipynb` |
| 2 | descriptive analysis | `notebooks/01_descriptive_analysis.ipynb` |
| 3 | baseline | `src/train_tfidf_baseline.py`, then `notebooks/02_baselines.ipynb` |
| 4 | Transformer fine-tuning | `src/finetune_transformer.py` (about 4 hours on a 4 GB GPU), `src/evaluate_transformer.py`, then `notebooks/03_transfer_learning_finetuning.ipynb` |
| 5 | predict the unlabeled pool | `src/predict_pseudo.py` |
| 6 | select pseudo-labels | `src/make_pseudo_labels.py` (threshold 0.90) |
| 7 | control and semi-supervised runs | `src/finetune_transformer.py` with `--init-from` and `--extra-train-file` (see the `args` in `reports/ctrl_200k_history.json` and `reports/ssl_200k_v2_history.json`) |
| 8 | test evaluation | `src/evaluate_test.py` (each model once) |
| 9 | comparison | `src/compare_test.py` (paired bootstrap, 2000 draws), then `notebooks/04_semi_supervised.ipynb` |
| 10 | error analysis | `src/export_errors.py` |
| 11 | offline demo | `src/predict_review.py` |

The notebooks `02`, `03` and `04` read saved results and finish in minutes; they do not retrain.
Do not re-run the training scripts with their default output folders on a finished project: they would overwrite the final artifacts.

## 5. Offline prediction for one review

```
python src/predict_review.py --text "کیفیت عالی بود و پیشنهاد می‌کنم"
python src/predict_review.py --file reviews.txt
python src/predict_review.py --demo --json reports/demo_predictions.json
```

It needs only saved files (`artifacts/tfidf_word_*.joblib` and `artifacts/ft_200k/`) and no internet.
It prints the prediction and probabilities of both models and a short explanation: the strongest TF-IDF terms for the baseline,
and the effect of removing words (occlusion) for the Transformer. For very confident predictions the occlusion scores are close to zero,
because the probabilities are saturated; do not read this as "no word matters".

## 5b. HTTP API and Docker image (optional part)

A FastAPI service (`app/main.py`) serves the saved TF-IDF model and the fine-tuned Transformer. It reuses the functions of
`src/predict_review.py` and `src/text_normalize.py`, so the API and the offline script give the same predictions.
Both models are loaded once at start-up. The only input is the review text; a rating or a label in the request is rejected.
Raw review texts are never written to the log. The maximum text length is 5000 characters (environment variable `MAX_TEXT_CHARS`).

### Endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` | The process is alive (always 200). |
| `GET /ready` | 200 when both models are loaded, 503 otherwise; also reports `load_seconds`. |
| `POST /normalize` | Returns the normalized text. |
| `POST /tokenize` | Transformer tokens, ids, token count, `truncated` flag. |
| `POST /predict/tfidf`, `POST /predict/transformer`, `POST /predict/compare` | Recommendation and rating-band labels with probabilities. |
| `POST /predict` | Same, with the model chosen in the body (`transformer`, `tfidf`, `compare`). |

Error codes: 422 for empty, whitespace-only or too long text and for extra fields; 400 for an unknown model or tokenizer;
503 when a model is not loaded; 500 for an unexpected error.

### Run without Docker

From the project root, with the environment active and the files of `artifacts/` present:

```
uvicorn app.main:app --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the endpoints. Example request:

```
curl -X POST http://127.0.0.1:8000/predict/compare -H "Content-Type: application/json" -d '{"text": "..."}'
```

### Tests

```
python -m pytest tests -v
```

There are 12 tests with `TestClient` (health, ready, normalize, 422 / 400 / 503 cases, one shape check per predict endpoint).
Result on the author's machine: 12 passed in 14.02 s (Python 3.11.18, pytest 9.2.0).

### Docker image (CPU only)

The image contains the code and the packages of `requirements-api.txt` (CPU build of torch). The model files are **not** inside
the image; the `artifacts/` folder is mounted read-only at `/artifacts`.

```
docker build -t digikala-api .
docker run --rm -p 8000:8000 -v "$(pwd)/artifacts:/artifacts:ro" --name digikala-api digikala-api
```

The Dockerfile uses the Aliyun mirrors for `pip` by default (the normal package servers were unreachable on the author's network).
On another network override them with `--build-arg INDEX_URL=https://pypi.org/simple/` and
`--build-arg TORCH_LINKS=https://download.pytorch.org/whl/cpu` (not tested here).

Measured on the author's machine (Ubuntu, CPU, Docker 29.7.2; single measurements, not averages):

| Item | Value |
|---|---|
| Build from scratch | 6 min 15 s |
| Image size (`docker images`) | disk usage 1.95 GB, content size 416 MB |
| Model loading (`load_seconds` from `/ready`) | 10.8 s in the container, 10.9 s without Docker |
| `/health` and `/ready` from inside the container | both answered, ready = true |
| One `/predict/compare` request (12-token review) | TF-IDF 3.1 ms, Transformer 36.1 ms |

Failure test: starting the container without mounting `artifacts/` gives `FileNotFoundError` for both models in the log;
the service stays up, `/health` returns 200, `/ready` and the predict endpoints return 503.

Assumptions and not tested: the models were trained with CUDA, the image uses the CPU build of torch; GPU in the container,
other networks, low-memory machines, wrong working directory and port conflicts were not tested.
## 6. Files that are not in git

| path | why | how to rebuild |
|---|---|---|
| `data/` | raw data and large processed files | place the raw CSV files, run notebook 00 |
| `artifacts/` | trained models (`best.pt` is about 650 MB), pool probabilities | steps 3 to 7 above |
| `reports/test_results/*_predictions.parquet` | per-row test predictions | `src/evaluate_test.py` |
| `reports/speed_test*`, `reports/resume_test*`, `reports/transformer_run_history.json` | small test logs of engineering experiments | not needed |

Because `artifacts/` is not in git, `notebooks/04_semi_supervised.ipynb` skips its pool-audit cells when the pool files are missing; the result tables still run.

## 7. Results (fixed random test sample of 99,635 rows)

Recommendation n = 85,279 and rating band n = 91,578 (rows with a valid target); the same rows are used for every model.
Source: `reports/test_results/*.json`; confidence intervals come from `reports/test_results/comparison.json`.

| model | rec macro F1 [95% CI] | rec accuracy | band macro F1 [95% CI] | band accuracy |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.6728 [0.6687, 0.6769] | 0.8073 | 0.6364 [0.6327, 0.6404] | 0.7263 |
| Transformer, supervised | 0.6933 [0.6891, 0.6976] | 0.8462 | 0.6584 [0.6548, 0.6624] | 0.7571 |
| Control (continued training, labeled rows only) | 0.6912 | 0.8416 | 0.6560 | 0.7543 |
| Semi-supervised v2 | 0.6963 [0.6920, 0.7006] | 0.8319 | 0.6630 [0.6591, 0.6668] | 0.7540 |
| Semi-supervised v1 (combined class weights; a mistake, kept as a sensitivity result) | 0.6878 | 0.8376 | 0.6518 | 0.7837 |

Findings, in short:

* Fine-tuning the Transformer is better than the baseline in macro F1 and accuracy on both heads. The balanced accuracy of the recommendation head is **not** clearly better.
* The semi-supervised run (v2) has a small macro-F1 gain over the control (+0.0051 recommendation, +0.0071 rating band; both intervals exclude 0), but the accuracy of the recommendation head drops (-0.0098) and its calibration error rises. One training run per setting and validation noise of about 0.01 mean the gain is not clearly robust.
* The class `no_idea` stays the hardest class for every model.

## 8. Seeds and fixed settings

| name | value |
|---|---|
| `SPLIT_SEED` | 2026 |
| `MASK_SEED` | `digikala-semi-supervised-v1` |
| `MASK_RATE` | 0.20 |
| training subset, pseudo-label cap, bootstrap | seed 2026 |
| pseudo-label threshold | 0.90 (recommendation head) |

## 9. Limitations

* The Transformer was fine-tuned on 200,000 training rows (about 6% of the labeled rows), with the word-embedding table frozen and `max_length = 64` (about 1.3% of reviews are truncated).
* One training run per setting; the bootstrap intervals describe test-sampling noise only.
* The recommendation label is partly subjective (the same text appears with different labels), which limits achievable accuracy.
* Colloquial negation and hedged phrases (for example "بد نیست") are still misclassified in some cases.
* The semi-supervised pool is artificial (hidden labels of training rows), so it follows the same distribution as the labeled data.
* Emoji are kept as they are (no normalisation rule removes them); HTML tags do not occur in the data.

## 10. Use of AI assistance

An AI assistant (Claude, by Anthropic) helped with parts of the code, the notebooks and the drafts of the text.
All experiments were run on the author's own machine, and all numbers in this repository come from the saved result files.
