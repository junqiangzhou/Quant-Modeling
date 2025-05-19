from data.data_fetcher import create_dataset
from data.utils import get_date_back
from config.config import (MODEL_EXPORT_NAME, ENCODER_TYPE, device,
                           look_back_window, label_names, buy_sell_signals,
                           feature_names)
from feature.label import compute_labels
from feature.feature import create_batch_feature
from model.model import PredictionModel, CustomLoss
from model.utils import StockDataset
from model.eval import eval_model

import pandas as pd
import pytest
import numpy as np
import torch


@pytest.fixture
def mock_fetch_stocks():
    # Mock fetch_stocks to return a small list of stock symbols
    return ["AAPL", "MSFT"]


def test_model_integration(mock_fetch_stocks):
    # Use only mock_fetch_stocks as the dependency
    start_date = "2023-01-01"
    end_date = "2023-03-31"

    # shift the start date back to get more data for history features
    shifted_start_date = get_date_back(start_date, look_back_window + 20)

    all_features, all_labels, all_dates = None, None, None
    for i, stock in enumerate(mock_fetch_stocks):
        print(">>>>>>stock: ", stock)
        try:
            df = create_dataset(stock, shifted_start_date, end_date)
            # create labels and add them into the dataframe
            df, _ = compute_labels(df)
            if df is None:
                continue

            # print("total # of data samples: ", df.shape[0])
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
    print("total # of feature samples: ", all_features.shape[0])

    test_dataset = StockDataset(all_features, all_labels)

    model = PredictionModel(feature_len=all_features.shape[2],
                            seq_len=all_features.shape[1],
                            encoder_type=ENCODER_TYPE).to(device)
    model.load_state_dict(
        torch.load(f"./model/export/{MODEL_EXPORT_NAME}.pth"))
    model.eval()
    criterion = CustomLoss()

    predict_probs, predict_labels = eval_model(model, criterion, test_dataset)
    assert predict_probs is not None, "Prediction probabilities should not be None"
    assert predict_labels is not None, "Prediction labels should not be None"
    assert len(predict_probs) == len(
        test_dataset), "Prediction probabilities length mismatch"
    assert len(predict_labels) == len(
        test_dataset), "Prediction labels length mismatch"
    assert predict_probs.shape[1] == len(
        label_names) * 3, "Prediction probabilities shape mismatch"
    assert predict_labels.shape[1] == len(
        label_names), "Prediction labels shape mismatch"

    assert all(np.isin(predict_labels.flatten(),
                       [0, 1, 2])), "Prediction labels should be in {0, 1, 2}"

    # Check the distribution of labels in each label
    assert (predict_labels[:, 0] == 0).sum(axis=0) == 77
    assert (predict_labels[:, 0] == 1).sum(axis=0) == 21
    assert (predict_labels[:, 0] == 2).sum(axis=0) == 10

    assert (predict_labels[:, -1] == 0).sum(axis=0) == 84
    assert (predict_labels[:, -1] == 1).sum(axis=0) == 19
    assert (predict_labels[:, -1] == 2).sum(axis=0) == 5
