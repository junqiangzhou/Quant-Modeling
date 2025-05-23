import pandas as pd
import pytest
from datetime import datetime
import os

from data.data_fetcher import create_dataset
from feature.label import compute_labels
from feature.feature import compute_online_feature, create_batch_feature
from data.utils import save_to_csv, load_from_csv, normalize_date


@pytest.fixture
def mock_fetch_data():
    # Mock fetch_stocks to return a small list of stock symbols
    stock = "AAPL"
    # Use only mock_fetch_stocks as the dependency
    start_date = "2023-01-01"
    end_date = "2023-03-31"

    # Simulate create_dataset behavior
    csv_file = f"./feature/test/test_{stock}_{start_date}_{end_date}.csv"
    if not os.path.exists(csv_file):
        df = create_dataset(stock, start_date, end_date)
        save_to_csv(df, csv_file)
    else:
        df = load_from_csv(csv_file)

    df, _ = compute_labels(df)
    return df


def test_create_batch_feature(mock_fetch_data):
    df = mock_fetch_data
    # Test the main integration of the feature computation
    features, labels, dates = create_batch_feature(df)

    # Check if the labels are computed correctly
    assert features.shape == (6, 50, 33
                              )  # 6 samples, 50 timesteps, 33 features
    assert labels.shape == (6, 4)  # 6 samples, 4 labels
    assert dates.shape == (6, )  # 6 samples


def test_compute_online_feature(mock_fetch_data):
    df = mock_fetch_data

    date = normalize_date("2023-01-20")
    features_scaled = compute_online_feature(df, date)
    # Check if the features are computed correctly
    assert features_scaled is None

    date = normalize_date("2023-03-24")
    features_scaled = compute_online_feature(df, date)
    # Check if the features are computed correctly
    assert features_scaled is not None
    assert features_scaled.shape == (1, 50, 33
                                     )  # 1 sample, 50 timesteps, 33 features
