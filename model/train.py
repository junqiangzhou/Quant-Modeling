import sys
import os

sys.path.append(os.path.abspath('..'))

from data.data_fetcher import get_stock_df
from feature.feature import create_batch_feature
from model.utils import check_nan_in_tensor, check_inf_in_tensor, StockDataset

from model.model import PredictionModel, CustomLoss, XGBoostClassifier
from typing import Tuple
import pandas as pd
import numpy as np
import random
from collections import Counter

import torch
import torch.nn as nn

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data import label
from config.config import ModelType, ENCODER_TYPE, MODEL_TYPE, device, random_seed
from model.eval import eval_model, eval_xgboost_model

# # 下载AAPL一年的股票数据
# df = yf.download('AAPL', start='2023-01-01', end='2024-01-01', interval='1d')

# # 计算百分比变化
# df_pct_change = df[['Open', 'High', 'Low', 'Close']].pct_change().dropna()

# # 增加交易量等特征（可以选择是否包含交易量特征）
# df_pct_change['Volume'] = df['Volume'].iloc[1:].values

# # 归一化数据
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(df_pct_change.values)

label_names = label.label_feature


def set_seed(seed=random_seed):
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you use multiple GPUs

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def multi_label_random_downsample(X, y, random_state=random_seed):
    """
    Apply RandomUnderSampler for multi-label data by considering each unique label combination as a separate class.

    Args:
        X (numpy.ndarray): Feature matrix (n_samples, seq_len, n_features)
        y (numpy.ndarray): Multi-label binary matrix (n_samples, n_labels)
        random_state (int): Random state for reproducibility

    Returns:
        X_resampled, y_resampled: Resampled feature matrix and label matrix
    """
    # Reshape X from 3D to 2D
    n_samples, seq_len, n_features = X.shape
    X_reshaped = X.reshape(n_samples, seq_len * n_features)

    # Convert multi-label matrix to label combinations
    y_combinations = [tuple(label) for label in y]

    # Count the occurrences of each combination
    combination_counts = Counter(y_combinations)
    zero_label = (0.0, ) * len(label_names)
    freq_sum = sum(
        [v for k, v in combination_counts.items() if k != zero_label])
    freq_zero = combination_counts[zero_label]
    # print(combination_counts)

    # RandomUnderSampler requires class labels
    y_comb_class = np.array([hash(label) for label in y_combinations])

    sampling_strategy = {}
    for k, v in combination_counts.items():
        if k == zero_label:
            sampling_strategy[hash(k)] = min(freq_sum, freq_zero)
        else:
            sampling_strategy[hash(k)] = v

    # Apply RandomUnderSampler
    ros = RandomUnderSampler(sampling_strategy=sampling_strategy,
                             random_state=random_state)

    X_resampled, y_resampled_comb = ros.fit_resample(X_reshaped, y_comb_class)

    # Recover original multi-label format
    hash_to_label = {hash(label): label for label in set(y_combinations)}
    y_resampled = np.array([hash_to_label[h] for h in y_resampled_comb])
    # y_resampled_combinations = [tuple(label) for label in y_resampled]
    # print(Counter(y_resampled_combinations))

    # Reshape X_resampled back to 3D
    X_resampled = X_resampled.reshape(-1, seq_len, n_features)

    print(f"sample size before: {X.shape[0]}, after: {X_resampled.shape[0]}")
    return X_resampled, y_resampled


def multi_label_random_oversample(X, y, random_state=random_seed):
    """
    Apply RandomOverSampler for multi-label data by considering each unique label combination as a separate class.

    Args:
        X (numpy.ndarray): Feature matrix (n_samples, seq_len, n_features)
        y (numpy.ndarray): Multi-label binary matrix (n_samples, n_labels)
        random_state (int): Random state for reproducibility

    Returns:
        X_resampled, y_resampled: Resampled feature matrix and label matrix
    """
    # Reshape X from 3D to 2D
    n_samples, seq_len, n_features = X.shape
    X_reshaped = X.reshape(n_samples, seq_len * n_features)

    # Convert multi-label matrix to label combinations
    y_combinations = [tuple(label) for label in y]

    # Count the occurrences of each combination
    combination_counts = Counter(y_combinations)

    # RandomOverSampler requires class labels
    y_comb_class = np.array([hash(label) for label in y_combinations])

    # Apply RandomOverSampler
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled_comb = ros.fit_resample(X_reshaped, y_comb_class)

    # Recover original multi-label format
    hash_to_label = {hash(label): label for label in set(y_combinations)}
    y_resampled = np.array([hash_to_label[h] for h in y_resampled_comb])

    # Reshape X_resampled back to 3D
    X_resampled = X_resampled.reshape(-1, seq_len, n_features)

    return X_resampled, y_resampled


def split_train_test_data(
        features: pd.DataFrame,
        labels: pd.DataFrame,
        batch_size=32) -> Tuple[DataLoader, StockDataset, np.ndarray]:
    # Assume features.shape = (n_samples, seq_len, n_features)
    # labels.shape = (n_samples, n_labels)
    indices = np.arange(
        features.shape[0])  # Create an array of original indices

    # Split data along with indices
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        features,
        labels,
        indices,
        test_size=0.2,
        random_state=random_seed,
        stratify=None,
        shuffle=True)

    # Oversampling makes testing worse, need to revisit
    # X_train, y_train = multi_label_random_oversample(X_train, y_train, random_state=random_seed)
    # Downsampling
    X_train, y_train = multi_label_random_downsample(X_train,
                                                     y_train,
                                                     random_state=random_seed)
    X_test, y_test = multi_label_random_downsample(X_test,
                                                   y_test,
                                                   random_state=random_seed)

    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    return train_loader, test_dataset, idx_test


def train_model(train_loader: DataLoader,
                epochs=10,
                learning_rate=0.001) -> Tuple[PredictionModel, CustomLoss]:
    # Set seed before model/training
    set_seed(random_seed)
    model = PredictionModel(feature_len=features.shape[2],
                            seq_len=features.shape[1],
                            encoder_type=ENCODER_TYPE).to(device)
    criterion = CustomLoss()
    # L2 regularization (weight decay)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-4)

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Train Neural Network
    epochs = epochs
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            check_nan_in_tensor(inputs)
            check_inf_in_tensor(inputs)
            check_nan_in_tensor(targets)
            check_inf_in_tensor(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=1.0)  # gradient clipping

            optimizer.step()

        if (epoch + 1) % 10 == 0:  # print loss every 10 epochs
            # predict_probs, predicted_labels = eval_model(
            # model, criterion, test_dataset)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model, criterion


def train_xgboost_model(train_loader: DataLoader) -> XGBoostClassifier:

    model = XGBoostClassifier(num_classes=len(label_names))

    # Train xGBoost
    X_train, y_train = None, None
    for inputs, targets in train_loader:
        check_nan_in_tensor(inputs)
        check_inf_in_tensor(inputs)
        check_nan_in_tensor(targets)
        check_inf_in_tensor(targets)

        if X_train is None:
            X_train, y_train = inputs, targets
        else:
            X_train, y_train = np.concatenate(
                (X_train, inputs), axis=0), np.concatenate((y_train, targets),
                                                           axis=0)

    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    csv_file = "data/stock_training_2023-01-01_2024-12-31.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Please run data_fetcher.py to download the data first.")
    else:
        df_all = pd.read_csv(csv_file)
        df_all['Date'] = pd.to_datetime(df_all['Date'])
        df_all.set_index('Date', inplace=True)
        df_all.index = df_all.index.date

    stocks = df_all['stock'].unique()
    all_features, all_labels, all_dates = None, None, None
    for i, stock in enumerate(stocks):
        print(">>>>>>stock: ", stock)
        try:
            df = get_stock_df(df_all, stock)
            features, labels, dates = create_batch_feature(df)
            if np.isnan(features).any() or np.isnan(labels).any():
                print(f"NaN detected in {stock}")
                continue
            if np.isinf(features).any() or np.isinf(labels).any():
                print(f"INF detected in {stock}")
                continue
        except:
            print(f"Error in processing {stock}")
            continue
        if all_features is None:
            all_features, all_labels, all_dates = features, labels, dates
        else:
            all_features = np.concatenate((all_features, features), axis=0)
            all_labels = np.concatenate((all_labels, labels), axis=0)
            all_dates = np.concatenate((all_dates, dates))
    print("total # of data samples: ", all_features.shape[0])

    train_loader, test_dataset, idx_test = split_train_test_data(
        all_features, all_labels, batch_size=128)

    if MODEL_TYPE == ModelType.TORCH:
        model, criterion = train_model(train_loader,
                                       epochs=150,
                                       learning_rate=1e-4)
        total_params = sum(p.numel() for p in model.parameters())
        print("total # of model params: ", total_params)
    else:
        model = train_xgboost_model(train_loader)

    # Eval model performance
    if MODEL_TYPE == ModelType.TORCH:
        predict_probs, predicted_labels = eval_model(model, criterion,
                                                     test_dataset)
    else:
        predict_probs, predicted_labels = eval_xgboost_model(
            model, test_dataset)

    if MODEL_TYPE == ModelType.TORCH:
        torch.save(model.state_dict(), './model/model.pth')
