"""Train and evaluate one of the forecasting models from the terminal."""

from __future__ import annotations

import argparse

from tensorflow.keras.callbacks import EarlyStopping

from src.data import ROOT, load_prices
from src.evaluate import (
    compute_metrics,
    naive_baseline,
    plot_history,
    plot_predictions,
    upsert_result,
)
from src.features import WINDOW, prepare_univariate
from src.models import build_rnn, set_seeds

MODELS_DIR = ROOT / "models"
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10
VALIDATION_SPLIT = 0.1


def main() -> None:
    """Run the selected model end to end and persist its metrics and artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["rnn"], default="rnn", help="model to train"
    )
    args = parser.parse_args()

    set_seeds()
    frame = load_prices()

    if args.model == "rnn":
        prepared = prepare_univariate(frame)
        model = build_rnn(WINDOW, n_features=1)
        name, label = "rnn", "SimpleRNN"
        scaler = prepared["scaler"]
    else:  # pragma: no cover - more choices are added in later phases
        raise ValueError(f"Unsupported model: {args.model}")

    early_stop = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    history = model.fit(
        prepared["X_train"],
        prepared["y_train"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=0,
    )
    epochs_run = len(history.history["loss"])

    predicted_scaled = model.predict(prepared["X_test"], verbose=0)
    actual = scaler.inverse_transform(
        prepared["y_test"].reshape(-1, 1)
    ).ravel()
    predicted = scaler.inverse_transform(predicted_scaled).ravel()
    assert len(actual) == len(prepared["test_dates"])

    metrics = compute_metrics(actual, predicted)
    baseline = naive_baseline(actual)

    history_path = plot_history(history, name)
    prediction_path = plot_predictions(prepared["test_dates"], actual, predicted, name)
    results_path = upsert_result({"Model": label, **metrics})
    upsert_result({"Model": "Naive baseline", **baseline})

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{name}.keras"
    model.save(model_path)

    print(f"model: {label} ({epochs_run} epochs)")
    print(
        f"test period: {prepared['test_dates'][0].date()} -> "
        f"{prepared['test_dates'][-1].date()} ({len(actual)} days)"
    )
    print(f"{label}: {metrics}")
    print(f"Naive baseline: {baseline}")
    print(f"saved: {history_path.relative_to(ROOT)}")
    print(f"saved: {prediction_path.relative_to(ROOT)}")
    print(f"saved: {model_path.relative_to(ROOT)}")
    print(f"updated: {results_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
