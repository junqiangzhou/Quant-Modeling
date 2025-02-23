import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import bisect

from datetime import datetime, timedelta

base_feature = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_20', 'MA_50'
]
macd_feature = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
feature_names = [name + "_diff" for name in base_feature
                 ] + [name + "_start" for name in base_feature] + macd_feature

# def normalize_features(features):
#     # Normalize numerical features
#     n_train_samples, n_timesteps, n_features = features.shape

#     # Reshape to 2D: (n_samples * n_timesteps, n_features)
#     features_reshaped = features.reshape(-1, n_features)

#     # Scale using StandardScaler
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features_reshaped)

#     # Reshape back to 3D: (n_samples, n_timesteps, n_features)
#     features_scaled = features_scaled.reshape(n_train_samples, n_timesteps,
#                                               n_features)

#     return features_scaled


def create_batch_feature(
        df: pd.DataFrame) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    look_back_window = 30

    start = look_back_window  # give some buffer as there could be NaN in df
    batch_list = []
    label_list = []
    date_list = []
    for i in range(start, len(df)):
        if pd.isna(df.iloc[i]["trend_5days+"]):
            continue

        history = df.iloc[i - look_back_window:i]
        history = history[feature_names]
        history = history.values
        batch_list.append(history)

        label = [
            df.iloc[i][name] for name in [
                'trend_5days+', 'trend_5days-', 'trend_10days+',
                'trend_10days-', 'trend_30days+', 'trend_30days-'
            ]
        ]
        label_list.append(label)
        date_list.append(df.index[i])

    features = np.stack(batch_list, axis=0)
    # features_scaled = normalize_features(features)
    labels = np.stack(label_list, axis=0)
    return features, labels, date_list


def compute_online_feature(df: pd.DataFrame,
                           date: datetime) -> NDArray[np.float64]:
    if date not in df.index:
        return None

    look_back_window = 30
    end_index = bisect.bisect_left(df.index, date) - 1
    start_index = end_index - look_back_window
    if start_index <= 0:
        return None

    history = df.iloc[start_index:end_index]
    history = history[feature_names]
    history = history.values
    return history
