from data.stocks_fetcher import fetch_stocks
from data.data_fetcher import create_dataset
from data.utils import get_date_back
from feature.feature import look_back_window, compute_online_feature
from model.model import PredictionModel
from config.config import (ENCODER_TYPE, label_feature, feature_names,
                           MODEL_EXPORT_NAME)

from datetime import date
import numpy as np
import torch
import time
import pandas as pd
import requests

stock1, stocks2 = fetch_stocks()
stocks = list(set(stock1 + stocks2))

# yahoo finance session
session = requests.Session()

# Load model
model = PredictionModel(feature_len=len(feature_names),
                        seq_len=look_back_window,
                        encoder_type=ENCODER_TYPE)
model.load_state_dict(torch.load(f"./model/export/{MODEL_EXPORT_NAME}.pth"))
model.eval()

# Initialize
buy_names = [name + "+" for name in label_feature]
sell_names = [name + "-" for name in label_feature]
hold_names = [name + "0" for name in label_feature]
columns = buy_names + sell_names + hold_names
pred = np.zeros((len(stocks), len(columns)))

for i, stock in enumerate(stocks):
    # avoid rate limit from Yahoo Finance
    if i % 100 == 0:
        time.sleep(60)

    today = date.today().strftime("%Y-%m-%d")
    start_date = get_date_back(today, look_back_window + 30)

    try:
        df = create_dataset(stock, start_date, today, session=session)
    except:
        print(f"{stock} data not available")
        continue

    last_date = df.index[-1]
    features = compute_online_feature(df, last_date)
    if features is None or np.isnan(features).any() or np.isinf(
            features).any():
        print(f"Features are not valid for {stock}")
    else:
        features_tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(features_tensor)
        logits = logits.reshape(len(label_feature), 3)
        probs = torch.softmax(
            logits, dim=1).float().numpy()  # convert logits to probabilities

        probs_hold = probs[:, 0]
        probs_buy = probs[:, 1]
        probs_sell = probs[:, 2]
        pred[i, :] = np.concatenate([probs_buy, probs_sell, probs_hold])
# Save results into a csv file
df_pred = pd.DataFrame(pred, index=stocks, columns=columns)
df_pred["BUY"] = df_pred[buy_names].mean(axis=1)
df_pred["SELL"] = df_pred[sell_names].mean(axis=1)
df_pred["HOLD"] = df_pred[hold_names].mean(axis=1)
df_pred = df_pred.sort_values(by="BUY", ascending=False)
df_pred.to_csv(f"./predict/data/prediction_{last_date}.csv",
               float_format="%.2f",
               index=True,
               index_label='stock')
