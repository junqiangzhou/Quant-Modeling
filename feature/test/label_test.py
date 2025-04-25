import pytest

from data.data_fetcher import create_dataset
from feature.label import compute_labels


@pytest.fixture
def mock_fetch_data():
    # Mock fetch_stocks to return a small list of stock symbols
    stock = "AAPL"
    # Use only mock_fetch_stocks as the dependency
    start_date = "2023-01-01"
    end_date = "2023-03-31"

    # Simulate create_dataset behavior
    df = create_dataset(stock, start_date, end_date)

    return df


def test_compute_labels(mock_fetch_data):
    # Test the main integration of the label computation
    df = mock_fetch_data
    df = compute_labels(df)

    # Check if the labels are computed correctly
    assert not df.empty
    assert len(df["stock"].unique()) == 1  # Two stocks: AAPL
    assert len(df) == 61  # Number of days between start_date and end_date
    assert len(df.columns) == 88
