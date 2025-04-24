from data.data_fetcher import create_dataset
from data.utils import get_date_back
from feature.label import compute_labels
from feature.feature import create_batch_feature
from model.utils import check_inf_in_tensor, check_nan_in_tensor, StockDataset
from model.model import PredictionModel, CustomLoss
from config.config import (ENCODER_TYPE, device, look_back_window,
                           label_feature, buy_sell_signals, feature_names)
from data.stocks_fetcher import MAG7, PICKS

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def eval_model(model, criterion, test_dataset):

    # Check buy/sell signals from features
    buy_signal_encoded = [signal + "_1" for signal in buy_sell_signals]
    buy_columns = [
        i for i, name in enumerate(feature_names) if name in buy_signal_encoded
    ]
    buy_features = test_dataset.X[:, -1, buy_columns]
    feature_has_buy = np.any(buy_features == 1, axis=1)
    feature_bullish = test_dataset.X[:, -1,
                                     feature_names.index("Price_Above_MA_5")]

    sell_signal_encoded = [signal + "_-1" for signal in buy_sell_signals]
    sell_columns = [
        i for i, name in enumerate(feature_names)
        if name in sell_signal_encoded
    ]
    sell_features = test_dataset.X[:, -1, sell_columns]
    feature_has_sell = np.any(sell_features == 1, axis=1)
    feature_bearish = test_dataset.X[:, -1,
                                     feature_names.index("Price_Below_MA_5")]

    # Model Evaluation
    model.eval()
    with torch.no_grad():
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

        logits = logits.reshape(targets.shape[0], len(label_feature), 3)
        probs, preds = None, None
        for i, label_name in enumerate(label_feature):
            logit = logits[:, i, :]
            target = targets[:, i]

            prob = torch.softmax(
                logit,
                dim=1).float().cpu().numpy()  # convert logits to probabilities

            pred = np.argmax(prob, axis=1)  # raw predictions
            feature_mask = False
            if feature_mask:
                # Update predictions based on buy/sell signals
                for j in range(len(pred)):
                    if pred[j] == 1 and (not feature_has_buy[j]
                                         or feature_bullish[j] != 1):
                        pred[j] = 0
                    elif pred[j] == 2 and (not feature_has_sell[j]
                                           or feature_bearish[j] != 1):
                        pred[j] = 0

            cm = confusion_matrix(target.squeeze().cpu().numpy(),
                                  pred,
                                  normalize='true')
            print(f"{label_name} Confusion Matrix: \n {cm}")

            if probs is None:
                probs, preds = prob, pred
            else:
                probs = np.column_stack((probs, prob))
                preds = np.column_stack((preds, pred))

    return probs, preds


def eval_xgboost_model(model, test_dataset):
    # Model Evaluation

    inputs = test_dataset.X
    targets = test_dataset.y
    # # Check input data
    # check_nan_in_tensor(inputs)
    # check_nan_in_tensor(targets)

    # Check model prediction
    preds, probs = model.predict(inputs)
    # check_nan_in_tensor(probs)

    for i, label_name in enumerate(label_feature):
        pred = preds[:, i]
        target = targets[:, i]

        cm = confusion_matrix(target.squeeze(), pred, normalize='true')
        print(f"{label_name} Confusion Matrix: \n {cm}")

    return probs, preds


if __name__ == "__main__":
    start_date = "2023-01-01"
    end_date = "2024-12-31"

    # shift the start date back to get more data for history features
    shifted_start_date = get_date_back(start_date, look_back_window + 20)
    testing_stocks = PICKS

    all_features, all_labels, all_dates = None, None, None
    df_all_list = []
    samples_list = []
    for i, stock in enumerate(testing_stocks):
        print(">>>>>>stock: ", stock)
        try:
            df = create_dataset(stock, shifted_start_date, end_date)
            # create labels and add them into the dataframe
            df = compute_labels(df)
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

    predict_probs, predict_labels = eval_model(model, criterion, test_dataset)

    pred_label_names = [label + "_pred" for label in label_feature]
    prob_label_names = [
        label + str(i) + "_prob" for label in label_feature for i in range(3)
    ]
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

    df_all.to_csv(f"./data/dataset/stock_testing_{start_date}_{end_date}.csv",
                  index=True,
                  index_label="Date")
