import sys
import os

sys.path.append(os.path.abspath('..'))

from data.data_fetcher import create_dataset_with_labels
from feature.feature import create_batch_feature
from data.stocks_fetcher import fetch_stocks

from model.model import PredictionModel
from typing import Tuple
import pandas as pd
import numpy as np
import random
from collections import Counter
import itertools

import torch
import torch.nn as nn

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

# # 下载AAPL一年的股票数据
# df = yf.download('AAPL', start='2023-01-01', end='2024-01-01', interval='1d')

# # 计算百分比变化
# df_pct_change = df[['Open', 'High', 'Low', 'Close']].pct_change().dropna()

# # 增加交易量等特征（可以选择是否包含交易量特征）
# df_pct_change['Volume'] = df['Volume'].iloc[1:].values

# # 归一化数据
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(df_pct_change.values)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 42


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


class StockDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()
        class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(class_weights)

    def forward(self, logits, targets):
        loss = self.criterion(logits, targets)
        return loss


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
    zero_label = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    freq_max = max(
        [v for k, v in combination_counts.items() if k != zero_label])
    print(combination_counts)

    # RandomUnderSampler requires class labels
    y_comb_class = np.array([hash(label) for label in y_combinations])

    sampling_strategy = {}
    for k, v in combination_counts.items():
        if k == zero_label:
            sampling_strategy[hash(k)] = v  #freq_max
        else:
            sampling_strategy[hash(k)] = v

    # Apply RandomUnderSampler
    ros = RandomUnderSampler(sampling_strategy=sampling_strategy,
                             random_state=random_state)

    X_resampled, y_resampled_comb = ros.fit_resample(X_reshaped, y_comb_class)

    # Recover original multi-label format
    hash_to_label = {hash(label): label for label in set(y_combinations)}
    y_resampled = np.array([hash_to_label[h] for h in y_resampled_comb])
    y_resampled_combinations = [tuple(label) for label in y_resampled]
    print(Counter(y_resampled_combinations))

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


# def normalize_features(X_train, X_test):
#     # Normalize numerical features
#     n_train_samples, n_timesteps, n_features = X_train.shape
#     n_test_samples = X_test.shape[0]
#     # Reshape to 2D: (n_samples * n_timesteps, n_features)
#     X_train_reshaped = X_train.reshape(-1, n_features)
#     X_test_reshaped = X_test.reshape(-1, n_features)

#     # Scale using StandardScaler
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train_reshaped)
#     X_test_scaled = scaler.transform(X_test_reshaped) # transform uses the same parameber calculated

#     # Reshape back to 3D: (n_samples, n_timesteps, n_features)
#     X_train_scaled = X_train_scaled.reshape(n_train_samples, n_timesteps, n_features)
#     X_test_scaled = X_test_scaled.reshape(n_test_samples, n_timesteps, n_features)

#     return X_train_scaled, X_test_scaled


def split_train_test_data(
        features: pd.DataFrame,
        labels: pd.DataFrame,
        batch_size=32
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
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
        stratify=None)

    # We have already normalized features when creating the features, so skip normalization
    # X_train, X_test = normalize_features(X_train, X_test)
    # Oversampling makes testing worse, need to revisit
    # X_train, y_train = multi_label_random_oversample(X_train, y_train, random_state=random_seed)
    # Downsampling
    # X_train, y_train = multi_label_random_downsample(X_train, y_train, random_state=random_seed)

    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=X_test.shape[0],
                             shuffle=True)
    return train_loader, test_loader, idx_train, idx_test


def train_model(train_loader: DataLoader,
                latent_dim=16,
                hidden_dim=32,
                epochs=10,
                learning_rate=0.001) -> Tuple[PredictionModel, CustomLoss]:
    # Set seed before model/training
    set_seed(random_seed)
    model = PredictionModel(input_dim=features.shape[2],
                            seq_len=features.shape[1],
                            latent_dim=latent_dim,
                            hidden_dim=hidden_dim).to(device)
    criterion = CustomLoss()
    # L2 regularization (weight decay)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-4)

    # Train Neural Network
    epochs = epochs
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=1.0)  # gradient clipping

            optimizer.step()

        if (epoch + 1) % 10 == 0:  # print loss every 10 epochs
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model, criterion


def eval_model(model, criterion, test_loader, idx_test, dates):
    # Model Evaluation
    model.eval()
    with torch.no_grad():
        # m, n = labels.shape
        metrics = ["TP", "FP", "FN"]
        label_names = [
            'trend_5days+', 'trend_5days-', 'trend_10days+', 'trend_10days-',
            'trend_30days+', 'trend_30days-'
        ]
        n = len(label_names)

        names_metrics = [
            metric + name
            for metric, name in list(itertools.product(label_names, metrics))
        ]
        stats_count = [{metric: 0 for metric in metrics} for _ in range(n)]
        stats_date = {name: [] for name in names_metrics}
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if np.isnan(inputs.cpu().numpy()).any() or np.isnan(
                    targets.cpu().numpy()).any():
                print("NaN detected in inputs")
            logits = model(inputs)
            if np.isnan(logits.cpu().numpy()).any():
                print("NaN detected in logits")
            loss = criterion(logits, targets)
            print(f"Test Loss: {loss.item():.4f}")
            probs = torch.sigmoid(logits)  # convert logits to probabilities
            preds = (probs > 0.5).float().cpu().numpy()  # binary predictions

            for col in range(n):
                for row in range(targets.shape[0]):
                    index = idx_test[row]
                    if targets[row, col] == 1 and preds[row, col] == 1:
                        stats_count[col]["TP"] += 1
                        stats_date[label_names[col] + "TP"].append(
                            dates[index])
                    elif targets[row, col] == 0 and preds[row, col] == 1:
                        stats_count[col]["FP"] += 1
                        stats_date[label_names[col] + "FP"].append(
                            dates[index])
                    elif targets[row, col] == 1 and preds[row, col] == 0:
                        stats_count[col]["FN"] += 1
                        stats_date[label_names[col] + "FN"].append(
                            dates[index])
                stats_date[label_names[col] + "TP"].sort()
                stats_date[label_names[col] + "FP"].sort()
                stats_date[label_names[col] + "FN"].sort()

    pr = [[0.0] * n, [0.0] * n]
    for col in range(n):
        if stats_count[col]["TP"] + stats_count[col]["FP"] > 0:
            pr[0][col] = stats_count[col]["TP"] / float(
                stats_count[col]["TP"] + stats_count[col]["FP"])
        if stats_count[col]["TP"] + stats_count[col]["FN"] > 0:
            pr[1][col] = stats_count[col]["TP"] / float(
                stats_count[col]["TP"] + stats_count[col]["FN"])
    pr_table = pd.DataFrame(data=pr,
                            index=["Precision", "Recall"],
                            columns=label_names)
    print(pr_table)

    # Find the length of the longest sublist
    max_rows = max([len(value) for value in stats_date.values()])
    padded_array = np.array([
        dates + [None] * (max_rows - len(dates))
        for dates in stats_date.values()
    ]).transpose()
    dates_table = pd.DataFrame(data=padded_array, columns=names_metrics)
    return pr_table, dates_table


if __name__ == "__main__":
    csv_file = "data/stock_training_2023-01-01_2024-12-31.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Please run data_fetcher.py to download the data first.")
    else:
        df_all = pd.read_csv(csv_file)

    stocks = df_all['stock'].unique()
    for i, stock in enumerate(stocks):
        print(">>>>>>stock: ", stock)
        try:
            df = df_all[df_all['stock'] == stock]
            features, labels, dates = create_batch_feature(df)
            if np.isnan(features).any() or np.isnan(labels).any():
                print(f"NaN detected in {stock}")
                continue
        except:
            print(f"Error in processing {stock}")
            continue
        if i == 0:
            all_features, all_labels, all_dates = features, labels, dates
        else:
            all_features = np.concatenate((all_features, features), axis=0)
            all_labels = np.concatenate((all_labels, labels), axis=0)
            all_dates += dates
    print("total # of data samples: ", all_features.shape[0])

    train_loader, test_loader, idx_train, idx_test = split_train_test_data(
        all_features, all_labels, batch_size=128)
    model, criterion = train_model(train_loader,
                                   latent_dim=32,
                                   hidden_dim=16,
                                   epochs=200,
                                   learning_rate=1e-3)
    total_params = sum(p.numel() for p in model.parameters())
    print("total # of model params: ", total_params)
    pr_table, dates_table = eval_model(model, criterion, test_loader, idx_test,
                                       all_dates)
    torch.save(model.state_dict(), './model/model.pth')
