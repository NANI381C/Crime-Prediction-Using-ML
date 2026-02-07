# src/model_lstm.py

import numpy as np
import pandas as pd
import math
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Bidirectional
)

# Helper: create sliding window sequences
def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_and_forecast(df_city, window=3, epochs=60, forecast_steps=1):

    df_city = df_city.sort_values("Date").reset_index(drop=True)

    # --------------------------
    # SELECT FEATURES
    # --------------------------
    features = [
        "Crime_Count",
        "Lag1",
        "Lag2",
        "Lag3",
        "TrendIndex",
        "Month_sin",
        "Month_cos"
    ]

    data = df_city[features].astype(float).values

    # --------------------------
    # TIME-BASED SPLIT
    # --------------------------
    test_size = max(int(len(data) * 0.2), 6)
    train_len = len(data) - test_size

    train_data = data[:train_len]
    test_data = data[train_len:]

    # --------------------------
    # SCALERS
    # --------------------------
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_all = data
    y_all = data[:, 0].reshape(-1, 1)  # Crime_Count

    scaler_X.fit(train_data)
    scaler_y.fit(train_data[:, 0].reshape(-1, 1))

    X_scaled = scaler_X.transform(X_all)
    y_scaled = scaler_y.transform(y_all)

    # --------------------------
    # SEQUENCES
    # --------------------------
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window)

    split_idx = max(0, train_len - window)

    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # --------------------------
    # MODEL
    # --------------------------
    model = keras.Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(window, X_train.shape[2])),
        Dropout(0.2),

        Bidirectional(LSTM(64)),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    # TRAIN
    model.fit(X_train, y_train, epochs=epochs, batch_size=8, verbose=1)

    # --------------------------
    # PREDICT TEST
    # --------------------------
    pred_scaled = model.predict(X_test)
    pred = scaler_y.inverse_transform(pred_scaled).flatten()
    y_true = scaler_y.inverse_transform(y_test).flatten()

    # --------------------------
    # METRICS
    # --------------------------
    rmse = math.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)
    mape = np.mean(np.abs((y_true - pred) / (y_true + 1e-6))) * 100

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "test_true": y_true,
        "test_pred": pred
    }

    # --------------------------
    # FORECAST FUTURE
    # --------------------------
    next_forecasts = []
    seq = X_scaled[-window:].copy()

    for _ in range(forecast_steps):
        input_seq = seq.reshape(1, window, X_scaled.shape[1])

        pred_scaled = model.predict(input_seq)[0][0]
        pred_inv = scaler_y.inverse_transform([[pred_scaled]])[0][0]

        next_forecasts.append(pred_inv)

        # Build next feature vector based on predicted crime count
        next_row = seq[-1].copy()
        next_row[0] = pred_scaled  # update Crime_Count scaled

        seq = np.vstack([seq[1:], next_row])

    return model, scaler_X, scaler_y, metrics, next_forecasts
