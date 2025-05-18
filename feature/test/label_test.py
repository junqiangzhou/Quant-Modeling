import pytest
import os
import pandas as pd

from data.data_fetcher import create_dataset
from feature.label import compute_labels
from data.utils import save_to_csv, load_from_csv


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

    return df


def test_compute_labels(mock_fetch_data):
    # Test the main integration of the label computation
    df = mock_fetch_data
    df, _ = compute_labels(df)

    # Check if the labels are computed correctly
    assert not df.empty
    assert len(df["stock"].unique()) == 1  # Two stocks: AAPL
    assert len(df) == 61  # Number of days between start_date and end_date
    assert len(df.columns) == 94
