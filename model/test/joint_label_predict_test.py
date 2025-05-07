import pytest
import torch

from data.data_fetcher import create_dataset
from feature.feature import compute_online_feature
from model.joint_label_predict import JointLabelPredictor
from config.config import label_names


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


@pytest.mark.skip
def test_compute_labels(mock_fetch_data):
    # Test the main integration of the label computation
    df = mock_fetch_data

    last_date = df.index[-1]
    features = compute_online_feature(df, last_date)

    model = JointLabelPredictor()
    features_tensor = torch.tensor(features, dtype=torch.float32)
    probs = model.predict(features_tensor)

    # Check if the labels are computed correctly
    assert probs.shape[0] == len(label_names) * 2
    assert probs.shape[1] == 3
