import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import bisect

from datetime import datetime, timedelta
from data import data_fetcher, label

base_feature = data_fetcher.base_feature
label_feature = label.label_feature

look_back_window = 30
macd_feature = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
kdj_feature = ["STOCHk_14_3_3", "STOCHd_14_3_3", "J"]
rsi_feature = ["RSI_14"]
buy_sell_signals_encoded = label.buy_sell_signals_encoded
bullish_bearish_signals = ["Price_Above_MA_5", "Price_Below_MA_5"]
# Other crossover signals that are not in buy_sell_signals
other_crossover_signals = [
    "MA_5_10_Crossover_Signal", "MA_5_50_Crossover_Signal",
    "MA_10_20_Crossover_Signal", "MA_20_50_Crossover_Signal",
    "VWAP_Crossover_Signal"
]
feature_names = [
    name + "_diff" for name in base_feature
] + buy_sell_signals_encoded + bullish_bearish_signals + other_crossover_signals


def normalize_features(features):
    # Normalize numerical features
    n_train_samples, n_timesteps, n_features = features.shape

    # Reshape to 2D: (n_samples * n_timesteps, n_features)
    features_reshaped = features.reshape(-1, n_features)

    # Scale using StandardScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_reshaped)

    # Reshape back to 3D: (n_samples, n_timesteps, n_features)
    features_scaled = features_scaled.reshape(n_train_samples, n_timesteps,
                                              n_features)

    return features_scaled


def create_batch_feature(
    df: pd.DataFrame
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    batch_list = []
    label_list = []
    date_list = []
    for i in range(look_back_window, len(df)):
        if pd.isna(df.iloc[i][label_feature[0]]
                   ):  # Skip earnings day where we don't compute labels
            continue

        history = df.iloc[i - look_back_window + 1:i + 1]
        history = history[feature_names]
        history = history.values
        batch_list.append(history)

        label = [df.iloc[i][name] for name in label_feature]
        label_list.append(label)
        date_list.append(df.index[i])

    features = np.stack(batch_list, axis=0)
    features_scaled = normalize_features(features)
    labels = np.stack(label_list, axis=0)
    dates = np.array(date_list)

    # check how many positive labels
    ones_per_column = np.sum(labels == 1, axis=0)
    print(f"Positive labels: {ones_per_column}")

    return features_scaled, labels, dates


def compute_online_feature(df: pd.DataFrame,
                           date: datetime) -> NDArray[np.float64]:
    if date not in df.index:
        return None

    end_index = bisect.bisect_left(df.index, date) + 1
    start_index = end_index - look_back_window
    if start_index <= 0:
        return None

    history = df.iloc[start_index:end_index]
    history = history[feature_names]
    history = history.values
    if np.isnan(history).any() or np.isinf(history).any():
        return None
    features = np.expand_dims(history, axis=0)
    features_scaled = normalize_features(features)
    return features_scaled
