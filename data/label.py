import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# List of labels where the model is trained against and predicts at inference time
label_feature = [
    "trend_5days+", "trend_5days-", "trend_10days+", "trend_10days-",
    "trend_30days+", "trend_30days-"
]
time_windows = [5, 10, 30]  # number of next rows to consider


# Computes the labels, and here's the basic idea:
# 1. Find the next date of interest (5, 10, 30... ahead), before the next earning date
# 2. Compute the (min, max) price during this time window
# 3. Positive label: If min_price > curr_price or (max_price > curr_price * 1.1 and max_price comes before min_price)
# 4. Negative label: If max_price < curr_price or (min_price < curr_price * 0.92 and min_price comes before max_price)
def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    up_multiplier = 1 + 0.1  # >= 10% is up
    down_multiplier = 1 - 0.08  # <= 8% is down

    next_earning_date_generator = (index for index, row in df.iterrows()
                                   if row["Earnings_Date"])
    curr_date = df.index[0]
    labels = pd.DataFrame(columns=label_feature)
    while next_date := next(next_earning_date_generator, None):
        # print(f"Quarter bewteen: {curr_date} and {next_date}")
        # (???) somehow the earnings date is 1day ahead, so I need to start from index 2 here.

        # Iterate over all rows before next earning date
        df_window = df.loc[curr_date:next_date]
        for i in range(len(df_window)):
            # Get the index and row at position i
            curr_index, curr_close = df_window.index[i], df_window[
                "Close"].iloc[i]

            # Skip 1st and last row as it's close to earnings date
            if i == 0 or i > len(df_window) - 6:
                labels.loc[curr_index] = [
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                ]
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
                up, down = 0, 0
                if min_close > curr_close or (
                        max_close > curr_close * up_multiplier and max_index
                        < min_index):  # straight up or up comes first
                    up = 1
                    # print(f"Date: {curr_index}, Close: {curr_close}, >>>>>Up: Percent {(max_close - curr_close) / curr_close * 100}, Length {max_index - curr_index}")
                elif max_close < curr_close or (
                        min_close < curr_close * down_multiplier and min_index
                        < max_index):  # straight down or down comes first
                    down = 1
                    # print(f"Date: {curr_index}, Close: {curr_close}, <<<<<Down: Percent {(min_close - curr_close) / curr_close * 100}, Length {min_index - curr_index}")
                label += [up, down]
            labels.loc[curr_index] = label

        curr_date = next_date + timedelta(days=1)

    return labels
