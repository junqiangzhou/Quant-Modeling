from data.stocks_fetcher import fetch_stocks
from data.data_fetcher import get_date_back, download_data
from data.label import one_hot_encoder, label_feature
from feature.feature import look_back_window, compute_online_feature, feature_names
from model.model import PredictionModel
from config.config import ENCODER_TYPE
from datetime import date

import numpy as np
import torch
import time
import pandas as pd
import yfinance as yf
import requests

stock1, stocks2 = fetch_stocks()
stocks = stock1 + stocks2

# yahoo finance session
session = requests.Session()

# Load model
model = PredictionModel(feature_len=len(feature_names),
                        seq_len=look_back_window,
                        encoder_type=ENCODER_TYPE)
model.load_state_dict(torch.load('./model/model.pth'))
model.eval()

# Initialize
buy_names = [name + "+" for name in label_feature]
sell_names = [name + "-" for name in label_feature]
columns = buy_names + sell_names
pred = np.zeros((len(stocks), len(columns)))

for i, stock in enumerate(stocks):
    # avoid rate limit from Yahoo Finance
    if i % 100 == 0:
        time.sleep(60)

    today = date.today().strftime("%Y-%m-%d")
    start_date = get_date_back(today, look_back_window + 30)

    try:
        df = download_data(stock, start_date, today, session=session)
        # Trim data within the interested time window
        df = df.loc[start_date:today]
        df.index = df.index.date
        df = one_hot_encoder(df)
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

        probs_buy = probs[:, 1]
        probs_sell = probs[:, 2]
        pred[i, :] = np.concatenate([probs_buy, probs_sell])
# Save results into a csv file
df_pred = pd.DataFrame(pred, index=stocks, columns=columns)
df_pred["buy_prob"] = df_pred[buy_names].mean(axis=1)
df_pred["sell_prob"] = df_pred[sell_names].mean(axis=1)
df_pred = df_pred.sort_values(by="buy_prob", ascending=False)
df_pred.to_csv(f"./prediction/prediction_{last_date}.csv",
               float_format="%.2f",
               index=True,
               index_label='stock')
