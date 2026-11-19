# AGENTS.md — stock-prediction

## Structure
- Work from the repo root with `venv/bin/python`. Python 3.12.16 is a portable CPython under `.python/`; `docs/venv` is an obsolete 3.14 venv with only pip — ignore it.
- `docs/guia.md` is the readable Markdown extraction of the spec `docs/Stock_Price_Prediction_with_LSTM_and_RNN_Complete_Build.md` (a DOCX misnamed `.md`). Follow its phase order. Deliberate deviations: headless matplotlib saving to `reports/`, scripts in `src/` (no notebooks), plus directional accuracy and `reports/results.csv`.
- Layout: `data/` (cache CSV), `models/` (.keras), `reports/` (PNGs + results.csv), `src/` (data.py, features.py, models.py, evaluate.py, train.py).

## Environment / verification
- Install from pinned `requirements.txt` with `venv/bin/pip install -r requirements.txt`. `pandas-ta==0.4.71b0` is a beta that pins `numba==0.61.2` and forces `numpy==2.2.6`; don't upgrade numpy independently.
- No display: call `matplotlib.use("Agg")` and `plt.savefig()` into `reports/`; never `plt.show()`.
- No test runner, lint, CI, or build config exists. Verify by running scripts directly; don't invent test commands.
- Code comments in English; user-facing explanations in Portuguese (Portugal).

## Git
- Root and `docs/` are separate git repos; there is no `.gitmodules`. `docs/` is an embedded repo that root git cannot track (`git add docs/...` silently does nothing), so commits are made in root only and `docs/guia.md` stays unversioned.
- Local identity: `OpenCode Agent <opencode@localhost>`.
- Generated data, `.keras` models, PNGs, `venv/` and `.python/` are gitignored; `reports/results.csv` is tracked.

## Correctness rules from the guide (easy to violate)
- Split chronologically (first 80% train, last 20% test); never shuffle time series before splitting.
- Fit `MinMaxScaler` on training data only; inverse-transform predictions before computing metrics.
- A window ending at index `i` predicts day `i`: split at `split_idx - WINDOW` (univariate) or `split_m - WINDOW + 1` (multivariate with shifted target). Keep test targets out of training.
- Keras `validation_split` takes the tail before shuffling, so validation stays later in time.
- Always report the naive baseline; chart-accurate price forecasts are usually just a one-day lag.
