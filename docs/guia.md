# Stock Price Prediction with LSTM and RNN: Complete Build Guide

An end-to-end deep learning project in Python using TensorFlow/Keras, Yahoo Finance data (AAPL), and technical indicators (RSI, EMA).

## 1. Project Overview

Stock price prediction is one of the most popular applications of deep learning in finance. Markets are noisy and volatile, and classical statistical methods often miss the non-linear, time-dependent patterns in price data. Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks are designed for sequences, so they are a natural fit for forecasting.

This guide walks you through building a complete forecasting system:

- Download historical Apple (AAPL) data from Yahoo Finance.
- Scale and transform the data into overlapping windows.
- Build and train a SimpleRNN model.
- Build and train an LSTM model.
- Add technical indicators (RSI, EMA) for a multivariate LSTM.
- Evaluate everything fairly against a naive baseline.

Aim: develop a robust stock price prediction pipeline using RNN and LSTM networks, while learning the fundamentals of deep learning applied to financial time series.

Important note: this is an educational project. Stock prices are close to a random walk, and a model that looks accurate on a chart is not a trading strategy. Never risk real money based on this project without rigorous backtesting.

### Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.10+ |
| Deep learning | TensorFlow / Keras |
| Data | yfinance, pandas, pandas-datareader, NumPy |
| Indicators | pandas_ta |
| Statistics and utilities | statsmodels, scikit-learn |
| Visualization | Matplotlib |

### Dataset

Daily AAPL data from the Yahoo Finance API: Open, High, Low, Close, and Volume for each trading day.

## 2. Environment Setup

Create a virtual environment and install the dependencies.

```python
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install tensorflow yfinance pandas numpy matplotlib scikit-learn statsmodels pandas_ta pandas-datareader jupyter
```

Suggested folder structure:

```python
stock-prediction/
├── data/                 # cached CSV files
├── notebooks/
│   └── stock_lstm.ipynb
├── models/               # saved .keras files
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   └── evaluate.py
└── requirements.txt
```

Common imports used throughout the guide:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

## 3. Neural Network Basics (Warm-up)

Before touching time series, make sure you understand how a plain neural network is built and trained in Keras.

Key concepts

- Neuron: computes a weighted sum of inputs plus a bias, then applies an activation function.
- Layer: a group of neurons. Networks stack an input layer, hidden layers, and an output layer.
- Loss function: measures prediction error (we use Mean Squared Error for regression).
- Optimizer: updates weights to reduce the loss (we use Adam).
- Epoch / batch: one pass over the training data / the number of samples per weight update.

Mini example on synthetic data

```python
X = np.random.rand(1000, 3)
y = (2 * X[:, 0] + 0.5 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.05, 1000))

model = keras.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
print(model.evaluate(X, y, verbose=0))
```

If the loss drops towards the noise level, your Keras setup works.

### Why recurrent networks?

A standard network treats each input independently. Time series have order: today's price depends on previous days.

- RNN: carries a hidden state from one time step to the next, giving it short-term memory. Plain RNNs suffer from the vanishing gradient problem, so they struggle with long dependencies.
- LSTM: adds a cell state and three gates (forget, input, output) that decide what to keep, add, and expose. This lets it learn longer-term patterns.

## 4. Loading the Time Series Data

```python
TICKER = "AAPL"
df = yf.download(TICKER, start="2012-01-01", end="2026-11-01",
                 auto_adjust=True, progress=False)

# Recent yfinance versions return MultiIndex columns; flatten them
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna()
df.to_csv("data/aapl.csv")   # cache so you do not re-download every run
print(df.shape)
df.head()
```

Sanity checks

```python
print(df.isna().sum())
print(df.index.is_monotonic_increasing)
print(df.describe())
```

### Quick exploration

```python
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
df["Close"].plot(ax=ax[0], title=f"{TICKER} Close Price")
df["Volume"].plot(ax=ax[1], title="Volume", color="gray")
plt.tight_layout(); plt.show()
```

Optional stationarity check with statsmodels. Prices are normally non-stationary, which is one reason forecasting them is hard.

```python
from statsmodels.tsa.stattools import adfuller

print("Close p-value:", adfuller(df["Close"])[1])
print("Returns p-value:", adfuller(df["Close"].pct_change().dropna())[1])
```

## 5. Data Transformations

### 5.1 Chronological train/test split

Never shuffle time series before splitting. The test set must come after the training set.

```python
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]
print(len(train_df), len(test_df))
```

### 5.2 Feature scaling

Neural networks train better when inputs share a small range. Use MinMaxScaler and fit it on training data only. Fitting on the full dataset leaks future information into training.

```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[["Close"]])
scaled = scaler.transform(df[["Close"]])   # scale the full series with train statistics
```

One caveat: because AAPL trends upward, test prices may exceed the training maximum and be scaled above 1. That is acceptable for the network, but it is a real limitation of price-level forecasting.

### 5.3 Overlapping windows

A sequence model needs samples of shape (window_size, n_features). Each sample is a window of past days, and the label is the next day's price.

```python
WINDOW = 60

def make_windows(data, target_col, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i, target_col])
    return np.array(X), np.array(y)

X, y = make_windows(scaled, target_col=0, window=WINDOW)

# A window ending at index i predicts day i, so split by target index
split_w = split_idx - WINDOW
X_train, y_train = X[:split_w], y[:split_w]
X_test, y_test = X[split_w:], y[split_w:]
print(X_train.shape, X_test.shape)   # (samples, 60, 1)
```

This way the first test window uses the final days of the training period as context, but no test target appears in training.

## 6. Recurrent Neural Network (SimpleRNN)

### 6.1 Build the model

```python
def build_rnn(window, n_features):
    model = keras.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.SimpleRNN(50, return_sequences=True),
        layers.Dropout(0.2),
        layers.SimpleRNN(50),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

rnn = build_rnn(WINDOW, 1)
rnn.summary()
```

### 6.2 Train

```python
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history_rnn = rnn.fit(
    X_train, y_train,
    epochs=100, batch_size=32,
    validation_split=0.1,      # last 10% of training windows, kept in time order
    callbacks=[early_stop],
    verbose=1,
)
```

Keras takes the validation split from the end of the training array before shuffling, so validation data is always later in time than the training data.

### 6.3 Plot training curves

```python
def plot_history(h, title):
    plt.figure(figsize=(8, 4))
    plt.plot(h.history["loss"], label="train")
    plt.plot(h.history["val_loss"], label="validation")
    plt.title(title); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend()
    plt.show()

plot_history(history_rnn, "SimpleRNN loss")
```

### 6.4 Sequence generation and evaluation

Write one evaluation helper and reuse it for every model.

```python
def evaluate(model, X_test, y_test, scaler, name):
    pred_scaled = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(pred_scaled).ravel()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    print(f"{name}: RMSE={rmse:.2f}  MAE={mae:.2f}  MAPE={mape:.2f}%")
    return actual, pred, {"model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape}

actual, pred_rnn, res_rnn = evaluate(rnn, X_test, y_test, scaler, "SimpleRNN")
```

### 6.5 Add a naive baseline

A model is only useful if it beats "tomorrow's price equals today's price".

```python
naive_pred = actual[:-1]
naive_actual = actual[1:]
naive_rmse = np.sqrt(mean_squared_error(naive_actual, naive_pred))
print(f"Naive baseline RMSE: {naive_rmse:.2f}")
```

### 6.6 Visualize predictions

```python
test_dates = df.index[split_idx:]

def plot_predictions(dates, actual, pred, title):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, label="Actual")
    plt.plot(dates, pred, label="Predicted", alpha=0.8)
    plt.title(title); plt.legend(); plt.show()

plot_predictions(test_dates, actual, pred_rnn, "SimpleRNN: actual vs predicted")
```

### 6.7 Multi-step forecasting (optional)

To generate a sequence of future prices, feed each prediction back into the window (recursive forecasting). Errors compound quickly, so expect accuracy to drop with horizon.

```python
def forecast_future(model, last_window, steps):
    window = last_window.copy()          # shape (WINDOW, 1)
    out = []
    for _ in range(steps):
        nxt = model.predict(window[np.newaxis, ...], verbose=0)[0, 0]
        out.append(nxt)
        window = np.vstack([window[1:], [[nxt]]])
    return scaler.inverse_transform(np.array(out).reshape(-1, 1)).ravel()

future_5 = forecast_future(rnn, scaled[-WINDOW:], steps=5)
```

## 7. LSTM Model

The data pipeline is identical. Only the architecture changes.

### 7.1 Build

```python
def build_lstm(window, n_features, units=64):
    model = keras.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(units),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

lstm = build_lstm(WINDOW, 1)
lstm.summary()
```

### 7.2 Train

```python
history_lstm = lstm.fit(
    X_train, y_train,
    epochs=100, batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    verbose=1,
)
plot_history(history_lstm, "LSTM loss")
```

### 7.3 Evaluate

```python
actual, pred_lstm, res_lstm = evaluate(lstm, X_test, y_test, scaler, "LSTM")
plot_predictions(test_dates, actual, pred_lstm, "LSTM: actual vs predicted")
```

Expect the LSTM to track the trend somewhat better than the SimpleRNN, but note how closely both resemble a one-day lag of the actual curve. That is typical for price-level forecasting.

## 8. Multivariate Input with LSTM

So far the model sees only the Close price. Now give it more context: volume, RSI (momentum), and EMA (trend).

### 8.1 Create technical indicators

```python
df_feat = df.copy()
df_feat["RSI"] = ta.rsi(df_feat["Close"], length=14)
df_feat["EMA20"] = ta.ema(df_feat["Close"], length=20)
df_feat["EMA50"] = ta.ema(df_feat["Close"], length=50)

df_feat = df_feat.dropna()      # indicators produce NaNs at the start
FEATURES = ["Close", "Volume", "RSI", "EMA20", "EMA50"]
print(df_feat[FEATURES].tail())
```

- RSI (14): ranges from 0 to 100. Above about 70 suggests overbought, below about 30 oversold.
- EMA: a moving average that weights recent prices more heavily, smoothing noise while following the trend.

If pandas_ta fails to install on your Python version, compute them in plain pandas:

```python
def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)

df_feat["RSI"] = rsi(df_feat["Close"])
df_feat["EMA20"] = df_feat["Close"].ewm(span=20, adjust=False).mean()
```

### 8.2 Create labels

The label is the next day's close. Shifting by one row aligns today's features with tomorrow's target.

```python
df_feat["Target"] = df_feat["Close"].shift(-1)
df_feat = df_feat.dropna()
```

Optional classification label (up or down), if you want to extend the project:

```python
df_feat["Direction"] = (df_feat["Target"] > df_feat["Close"]).astype(int)
```

### 8.3 Split and scale

Use separate scalers for features and for the target, so you can invert the predictions later. Fit both on training data only.

```python
split_m = int(len(df_feat) * 0.8)
train_m, test_m = df_feat.iloc[:split_m], df_feat.iloc[split_m:]

feat_scaler = MinMaxScaler().fit(train_m[FEATURES])
target_scaler = MinMaxScaler().fit(train_m[["Target"]])

X_all = feat_scaler.transform(df_feat[FEATURES])
y_all = target_scaler.transform(df_feat[["Target"]]).ravel()
```

### 8.4 Windows for multivariate data

```python
def make_windows_multi(X_data, y_data, window):
    Xs, ys = [], []
    for i in range(window, len(X_data) + 1):
        Xs.append(X_data[i - window:i])
        ys.append(y_data[i - 1])        # target of the last day in the window
    return np.array(Xs), np.array(ys)

Xm, ym = make_windows_multi(X_all, y_all, WINDOW)

split_wm = split_m - WINDOW + 1
Xm_train, ym_train = Xm[:split_wm], ym[:split_wm]
Xm_test, ym_test = Xm[split_wm:], ym[split_wm:]
print(Xm_train.shape, Xm_test.shape)   # (samples, 60, 5)
```

### 8.5 Build, train, evaluate

```python
lstm_multi = build_lstm(WINDOW, len(FEATURES))

history_multi = lstm_multi.fit(
    Xm_train, ym_train,
    epochs=100, batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    verbose=1,
)
plot_history(history_multi, "Multivariate LSTM loss")

actual_m, pred_multi, res_multi = evaluate(
    lstm_multi, Xm_test, ym_test, target_scaler, "Multivariate LSTM")
plot_predictions(df_feat.index[split_m:], actual_m, pred_multi,
                 "Multivariate LSTM: actual vs predicted")
```

## 9. Compare the Models

```python
results = pd.DataFrame([res_rnn, res_lstm, res_multi]).set_index("model")
print(results.round(2))
results[["RMSE", "MAE"]].plot(kind="bar", figsize=(8, 4), title="Model comparison")
plt.show()
```

The multivariate models use a slightly different date range because the indicators drop the first rows. For a strict comparison, evaluate all models on the same test dates.

Save your best model:

```python
lstm_multi.save("models/lstm_multivariate.keras")
```

## 10. Pitfalls and Best Practices

- Data leakage: fit scalers on training data only, split chronologically, and never shuffle before splitting.
- Lag illusion: predictions that look great often just copy yesterday's value. Always compare to the naive baseline.
- Overfitting: use dropout, early stopping, and fewer units. Financial data has a low signal-to-noise ratio.
- Non-stationarity: consider predicting daily returns or log returns instead of price levels.
- Reproducibility: set seeds and record the library versions you used. Results will still vary slightly between runs.
- Metrics: RMSE and MAE measure price error. For trading relevance, also measure directional accuracy.
- Transaction costs and risk: a real strategy requires backtesting with costs, slippage, and walk-forward validation.

## 11. Hyperparameters to Experiment With

| Parameter | Try |
| --- | --- |
| Window size | 20, 30, 60, 90 |
| LSTM units | 32, 64, 128 |
| Layers | 1, 2, 3 |
| Dropout | 0.1 to 0.4 |
| Learning rate | 1e-2, 1e-3, 1e-4 |
| Batch size | 16, 32, 64 |
| Features | add MACD, Bollinger Bands, OBV |

## 12. Extensions

- Predict returns instead of prices.
- Add GRU and Bidirectional LSTM models.
- Use walk-forward (rolling) cross-validation.
- Compare with a classical ARIMA model from statsmodels.
- Predict multiple steps ahead in one shot (sequence-to-sequence).
- Add market-wide data (S&P 500, VIX) or news sentiment.
- Deploy the model with Streamlit or FastAPI for a live demo.

## 13. Final Checklist

- ☐ Environment installed and AAPL data downloaded
- ☐ Chronological split and scaling with no leakage
- ☐ Windowed arrays with correct shapes
- ☐ SimpleRNN trained and evaluated
- ☐ LSTM trained and evaluated
- ☐ RSI and EMA features created, multivariate LSTM trained
- ☐ All models compared against the naive baseline
- ☐ Plots, metrics, and conclusions documented
- ☐ Code cleaned up and pushed to GitHub with a README

Disclaimer: this project is for educational purposes only and is not financial advice.
