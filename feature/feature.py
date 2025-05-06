import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
import bisect

from datetime import datetime
from config.config import (label_names, look_back_window, feature_names)

verbose = 0


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
        if pd.isna(df.iloc[i][label_names[0]]
                   ):  # Skip earnings day where we don't compute labels
            continue

        history = df.iloc[i - look_back_window + 1:i + 1]
        history = history[feature_names]
        history = history.values

        feature = np.expand_dims(history, axis=0)
        feature_scaled = normalize_features(feature)
        batch_list.append(np.squeeze(feature_scaled, axis=0))

        label = [df.iloc[i][name] for name in label_names]
        label_list.append(label)
        date_list.append(df.index[i])

    features_scaled = np.stack(batch_list, axis=0)
    labels = np.stack(label_list, axis=0)
    dates = np.array(date_list)

    # check how many positive labels
    if verbose >= 1:
        buys_per_column = np.sum(labels == 1, axis=0)
        sells_per_column = np.sum(labels == 2, axis=0)
        print(
            f"Buy labels: {buys_per_column}, Sell labels: {sells_per_column}")

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
