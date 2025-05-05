import pandas as pd
import numpy as np
from datetime import timedelta

from data.utils import perc_change, days_diff
from config.config import (future_time_windows, label_debug_columns,
                           buy_sell_signals, LABEL_TYPE, LabelType)


def is_stock_trending_up(curr_close, max_close, max_index, min_close,
                         min_index, up_perc_threshold, down_perc_threshold):
    if min_close > curr_close:  # straight up
        return True
    # max exceeds up threshold, and min stays below down threshold
    if perc_change(curr_close, max_close) > up_perc_threshold and perc_change(
            curr_close, min_close) > down_perc_threshold:
        return True
    # max exceeds up threshold, and min exceeds down threshold, but max comes before min
    if perc_change(curr_close, max_close) > up_perc_threshold and perc_change(
            curr_close,
            min_close) < down_perc_threshold and max_index < min_index:
        return True
    return False


def is_stock_trending_down(curr_close, max_close, max_index, min_close,
                           min_index, up_perc_threshold, down_perc_threshold):
    if max_close < curr_close:  # straight down
        return True
    # min exceeds down threshold, and max stays below up threshold
    if perc_change(curr_close, max_close) < up_perc_threshold and perc_change(
            curr_close, min_close) < down_perc_threshold:
        return True
    # min exceeds down threshold, and max exceeds up threshold, but min comes before max
    if perc_change(curr_close, max_close) > up_perc_threshold and perc_change(
            curr_close,
            min_close) < down_perc_threshold and min_index < max_index:
        return True
    return False


# Computes the labels, and here's the basic idea:
# 1. Find the next date of interest (5, 10, 30... ahead), before the next earning date
# 2. Compute the (min, max) price during this time window
# 3. Positive label: If min_price > curr_price or (max_price > curr_price * 1.1 and max_price comes before min_price)
# 4. Negative label: If max_price < curr_price or (min_price < curr_price * 0.92 and min_price comes before max_price)
def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    daily_change_perc = np.percentile(np.abs(df["daily_change"]), 99)
    print(f"daily change percentile: {daily_change_perc: .2f}")
    next_earning_date_generator = (index for index, row in df.iterrows()
                                   if row["Earnings_Date"])
    curr_date = df.index[0]
    labels = pd.DataFrame(np.nan, columns=label_debug_columns, index=df.index)
    while curr_date < df.index[-1]:
        # print(f"Quarter between: {curr_date} and {next_date}")
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

            # Skip 1st and last 6 rows as it's close to earnings date
            if i == 0 or i > len(df_window) - 6:
                continue

            label = []
            for N in future_time_windows:
                up_perc_threshold = min(daily_change_perc * N * 0.2 * 2,
                                        0.5)  # >= 20% going up at daily perc
                down_perc_threshold = -min(
                    daily_change_perc * N * 0.16 * 2,
                    0.4)  # >= 16% going down at daily perc

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

                trend = 0
                if is_stock_trending_up(curr_close, max_close, max_index,
                                        min_close, min_index,
                                        up_perc_threshold,
                                        down_perc_threshold):
                    if LABEL_TYPE == LabelType.PRICE:
                        trend = 1
                    elif LABEL_TYPE == LabelType.TREND and any(
                            buy_sell_signals_vals ==
                            1) and bullish_signal == 1:
                        # print(
                        #     f"Buy signal, {curr_index.date().strftime('%Y-%m-%d')}"
                        # )
                        trend = 1  # 1 - trend up
                    # else:
                    # print(f"No indicator for buy signal, {curr_index.date().strftime('%Y-%m-%d')}")
                    # print(f"Date: {curr_index}, Close: {curr_close}, >>>>>Up: Percent {(max_close - curr_close) / curr_close * 100}, Length {max_index - curr_index}")
                elif is_stock_trending_down(curr_close, max_close, max_index,
                                            min_close, min_index,
                                            up_perc_threshold,
                                            down_perc_threshold):
                    if LABEL_TYPE == LabelType.PRICE:
                        trend = 2
                    elif LABEL_TYPE == LabelType.TREND and any(
                            buy_sell_signals_vals ==
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

    return df, daily_change_perc
