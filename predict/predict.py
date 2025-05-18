from data.stocks_fetcher import fetch_stocks
from data.data_fetcher import create_dataset
from data.utils import get_date_backk, save_to_csv
from feature.feature import compute_online_feature
from model.joint_label_predict import JointLabelPredictor

from config.config import (label_names, look_back_window)

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
model = JointLabelPredictor()

# Initialize
buy_names = [name + "+" for name in label_names] + [
    name.replace("trend", "price") + "+" for name in label_names
]
sell_names = [name + "-" for name in label_names] + [
    name.replace("trend", "price") + "-" for name in label_names
]
hold_names = [name + "0" for name in label_names] + [
    name.replace("trend", "price") + "0" for name in label_names
]
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
        probs = model.predict(features_tensor)

        probs_hold = probs[:, 0]
        probs_buy = probs[:, 1]
        probs_sell = probs[:, 2]
        pred[i, :] = np.concatenate([probs_buy, probs_sell, probs_hold])
# Save results into a csv file
df_pred = pd.DataFrame(pred, index=stocks, columns=columns)
# Calculate the mean of buy/sell/hold signals using only the trend labels
df_pred["BUY"] = df_pred[buy_names[:len(label_names)]].mean(axis=1)
df_pred["SELL"] = df_pred[sell_names[:len(label_names)]].mean(axis=1)
df_pred["HOLD"] = df_pred[hold_names[:len(label_names)]].mean(axis=1)
df_pred = df_pred.sort_values(by="BUY", ascending=False)
df_pred.to_csv(f"./predict/data/prediction_{last_date}.csv",
               float_format="%.2f",
               index=True,
               index_label='stock')
