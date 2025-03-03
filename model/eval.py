import sys
import os
import itertools
import torch
import pandas as pd
import numpy as np
import random

from data.data_fetcher import get_stock_df, create_dataset_with_labels
from feature.feature import create_batch_feature
from model.utils import check_inf_in_tensor, check_nan_in_tensor, StockDataset
from data import label
from model.model import PredictionModel, CustomLoss
from config.config import ENCODER_TYPE, device
from data.stocks_fetcher import fetch_stocks

label_names = label.label_feature


def eval_model(model, criterion, test_dataset, test_dates):
    # Model Evaluation
    model.eval()
    with torch.no_grad():
        # m, n = labels.shape
        metrics = ["TP", "FP", "FN"]
        n = len(label_names)

        names_metrics = [
            metric + name
            for metric, name in list(itertools.product(label_names, metrics))
        ]
        stats_count = [{metric: 0 for metric in metrics} for _ in range(n)]
        stats_date = {name: [] for name in names_metrics}

        inputs = torch.from_numpy(test_dataset.X).to(device)
        targets = torch.from_numpy(test_dataset.y).to(device)
        # Check input data
        check_nan_in_tensor(inputs)
        check_nan_in_tensor(targets)

        # Check model prediction
        logits = model(inputs)
        check_nan_in_tensor(logits)

        loss = criterion(logits, targets)
        print(f"Test Loss: {loss.item():.4f}")

        probs = torch.sigmoid(
            logits).float().cpu().numpy()  # convert logits to probabilities
        preds = probs > 0.5  # binary predictions

        for col in range(n):
            for row in range(targets.shape[0]):
                if targets[row, col] == 1 and preds[row, col] == 1:
                    stats_count[col]["TP"] += 1
                    stats_date[label_names[col] + "TP"].append(test_dates[row])
                elif targets[row, col] == 0 and preds[row, col] == 1:
                    stats_count[col]["FP"] += 1
                    stats_date[label_names[col] + "FP"].append(test_dates[row])
                elif targets[row, col] == 1 and preds[row, col] == 0:
                    stats_count[col]["FN"] += 1
                    stats_date[label_names[col] + "FN"].append(test_dates[row])
            stats_date[label_names[col] + "TP"].sort()
            stats_date[label_names[col] + "FP"].sort()
            stats_date[label_names[col] + "FN"].sort()

            # print(f"{label_names[col]} TP count: {stats_count[col]['TP']}")
            # print(f"{label_names[col]} FP count: {stats_count[col]['FP']}")
            # print(f"{label_names[col]} FN count: {stats_count[col]['FN']}")

    # calculate precision and recall metrics
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

    # Find the length of the longest sublist
    max_rows = max([len(value) for value in stats_date.values()])
    padded_array = np.array([
        dates + [None] * (max_rows - len(dates))
        for dates in stats_date.values()
    ]).transpose()
    dates_table = pd.DataFrame(data=padded_array, columns=names_metrics)
    return probs, preds, pr_table, dates_table


if __name__ == "__main__":
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    stocks = fetch_stocks()

    random_seed = 21
    random.seed(random_seed)
    stocks_testing = random.sample(stocks, 30)

    all_features, all_labels, all_dates = None, None, None
    df_all_list = []
    samples_list = []
    for i, stock in enumerate(stocks_testing[:5]):
        print(">>>>>>stock: ", stock)
        try:
            df = create_dataset_with_labels(stock,
                                            start_date,
                                            end_date,
                                            vis=False)

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

        df_all_list.append(df)
        samples_list.append(features.shape[0])

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
    model.load_state_dict(torch.load('./model/model.pth'))
    model.eval()
    criterion = CustomLoss()

    predict_probs, predict_labels, pr_table, dates_table = eval_model(
        model, criterion, test_dataset, all_dates)
    print(pr_table)
    # print(dates_table)

    pred_label_names = [label + "_pred" for label in label_names]
    prob_label_names = [label + "_prob" for label in label_names]
    count = 0
    df_all = None
    for i, df in enumerate(df_all_list):
        next_count = count + samples_list[i]
        predict_label, predict_prob, dates = predict_labels[
            count:next_count, :], predict_probs[
                count:next_count, :], all_dates[count:next_count]
        df_pred = pd.DataFrame(data=np.concatenate(
            (predict_label, predict_prob), axis=1),
                               columns=pred_label_names + prob_label_names,
                               index=dates)
        df = df.join(df_pred, how='left')
        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=False)
        count = next_count

    df_all.to_csv(f"./data/stock_testing_2023-01-01_2024-12-31.csv",
                  index=True,
                  index_label="Date")
