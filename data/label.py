import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import chain

# List of labels where the model is trained against and predicts at inference time
time_windows = [5, 10, 20, 30]  # number of next rows to consider
# classification labels for model to predict
label_feature = list(chain(*[[f"trend_{time}days"] for time in time_windows]))
# all columns added for labeling purpose
# [max_close, max_duration, min_close, min_duration, trend_Xdays+, trend_Xdays-]
label_columns = list(
    chain(*[[
        f"{time}days_max_close", f"{time}days_max_duration",
        f"{time}days_min_close", f"{time}days_min_duration",
        f"trend_{time}days"
    ] for time in time_windows]))

buy_sell_signals = [
    "MA_5_20_Crossover_Signal",  # "MA_5_10_Crossover_Signal", "MA_5_50_Crossover_Signal", 
    "MA_10_50_Crossover_Signal",  # "MA_10_20_Crossover_Signal", "MA_20_50_Crossover_Signal",
    "MACD_Crossover_Signal",
    "RSI_Over_Bought_Signal",
    "BB_Signal",  # "VWAP_Crossover_Signal"
]
buy_sell_signals_encoded = [
    f"{signal}_{suffix}" for signal in buy_sell_signals
    for suffix in ["0", "-1", "1"]
]


def perc_change(curr, future):
    return (future - curr) / max(abs(curr), 1e-3)


def days_diff(date1, date2):
    return abs((date2 - date1).days)


def one_hot_encoder(df: pd.DataFrame) -> pd.DataFrame:
    # one-hot encoding on buy_sell_signals
    df_dummies = pd.get_dummies(df,
                                columns=buy_sell_signals,
                                prefix={col: col
                                        for col in buy_sell_signals})
    # Fill missing category with 0s.
    for col in buy_sell_signals_encoded:
        if col not in df_dummies.columns:
            df_dummies[col] = 0

    df = df.join(df_dummies[buy_sell_signals_encoded])
    return df


# Computes the labels, and here's the basic idea:
# 1. Find the next date of interest (5, 10, 30... ahead), before the next earning date
# 2. Compute the (min, max) price during this time window
# 3. Positive label: If min_price > curr_price or (max_price > curr_price * 1.1 and max_price comes before min_price)
# 4. Negative label: If max_price < curr_price or (min_price < curr_price * 0.92 and min_price comes before max_price)
def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    up_perc_threshold = 0.05  # >= 10% is up
    down_perc_threshold = -0.04  # <= 8% is down

    next_earning_date_generator = (index for index, row in df.iterrows()
                                   if row["Earnings_Date"])
    curr_date = df.index[0]
    labels = pd.DataFrame(np.nan, columns=label_columns, index=df.index)
    while curr_date < df.index[-1]:
        # print(f"Quarter bewteen: {curr_date} and {next_date}")
        # (???) somehow the earnings date is 1day ahead, so I need to start from index 2 here.

        next_date = next(next_earning_date_generator, df.index[-1])
        # Iterate over all rows before next earning date
        df_window = df.loc[curr_date:next_date]
        for i in range(len(df_window)):
            # Get the index and row at position i
            curr_index, curr_close = df_window.index[i], df_window[
                "Close"].iloc[i]
            buy_sell_signals_vals = df_window.loc[curr_index,
                                                  buy_sell_signals].values
            bullish_signal = df_window.loc[curr_index, "Price_Above_MA_5"]
            bearish_signal = df_window.loc[curr_index, "Price_Below_MA_5"]

            # Skip 1st and last row as it's close to earnings date
            if i == 0 or i > len(df_window) - 6:
                continue

            label = []
            for N in time_windows:
                # Calculate slice for next N rows, clamp to end
                end_pos = min(i + N, len(df_window))
                next_rows = df_window.iloc[i + 1:end_pos]
                # print(f"time window: start {curr_index}, end {df_window.index[end_pos - 1]}")

                # Compute max and min of "Close"
                max_close, max_index = next_rows["Close"].max(
                ), next_rows["Close"].idxmax()
                min_close, min_index = next_rows["Close"].min(
                ), next_rows["Close"].idxmin()

                # print(f"Date: {curr_index}, Close: {curr_close}, Max Date: {max_index}, Close: {max_close}, Min Date: {min_index}, Close: {min_close},")
                def is_stock_trending_up(curr_close, max_close, max_index,
                                         min_close, min_index):
                    if min_close > curr_close:  # straight up
                        return True
                    # max exceeds up threshold, and min stays below down threshold
                    if perc_change(
                            curr_close,
                            max_close) > up_perc_threshold and perc_change(
                                curr_close, min_close) > down_perc_threshold:
                        return True
                    # max exceeds up threshold, and min exceeds down threshold, but max comes before min
                    if perc_change(
                            curr_close,
                            max_close) > up_perc_threshold and perc_change(
                                curr_close, min_close
                            ) < down_perc_threshold and max_index < min_index:
                        return True
                    return False

                def is_stock_trending_down(curr_close, max_close, max_index,
                                           min_close, min_index):
                    if max_close < curr_close:  # straight down
                        return True
                    # min exceeds down threshold, and max stays below up threshold
                    if perc_change(
                            curr_close,
                            max_close) < up_perc_threshold and perc_change(
                                curr_close, min_close) < down_perc_threshold:
                        return True
                    # min exceeds down threshold, and max exceeds up threshold, but min comes before max
                    if perc_change(
                            curr_close,
                            max_close) > up_perc_threshold and perc_change(
                                curr_close, min_close
                            ) < down_perc_threshold and min_index < max_index:
                        return True
                    return False

                trend = 0
                if is_stock_trending_up(curr_close, max_close, max_index,
                                        min_close, min_index):
                    if any(buy_sell_signals_vals == 1) and bullish_signal == 1:
                        trend = 1  # 1 - trend up
                        # print(
                        #     f"Buy signal, {curr_index.date().strftime('%Y-%m-%d')}"
                        # )
                    # else:
                    # print(f"No indicator for buy signal, {curr_index.date().strftime('%Y-%m-%d')}")
                    # print(f"Date: {curr_index}, Close: {curr_close}, >>>>>Up: Percent {(max_close - curr_close) / curr_close * 100}, Length {max_index - curr_index}")
                elif is_stock_trending_down(curr_close, max_close, max_index,
                                            min_close, min_index):
                    if any(buy_sell_signals_vals ==
                           -1) and bearish_signal == 1:
                        # print(
                        #     f"Sell signal, {curr_index.date().strftime('%Y-%m-%d')}"
                        # )
                        trend = 2  # 2 - trend down
                    # else:
                    # print(f"No indicator for sell signal", {curr_index.date().strftime('%Y-%m-%d')})
                    # print(f"Date: {curr_index}, Close: {curr_close}, <<<<<Down: Percent {(min_close - curr_close) / curr_close * 100}, Length {min_index - curr_index}")
                label += [
                    perc_change(curr_close, max_close),
                    days_diff(curr_index, max_index),
                    perc_change(curr_close, min_close),
                    days_diff(curr_index, min_index), trend
                ]
            labels.loc[curr_index] = label

        curr_date = next_date + timedelta(days=1)

    df = df.join(labels, how='right')
    df = df.iloc[1:]  # drop 1st row

    df = one_hot_encoder(df)
    return df
