import pandas as pd
from yfinance import Ticker
from yahoo_fin import stock_info as si
import numpy as np

from datetime import datetime, timedelta
from data.visualize import visualize_dataset
from data.indicator import add_macd, add_moving_averages
from data.label import compute_labels

# List of basic data downloaded from Yahoo Finance
base_feature = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_20', 'MA_50'
]


# Add columns that calculates the delta w.r.t. previous row for each base feature
# This normalizes data by calculating the percentage change
# Returns updated dataframe
def add_delta_from_prev_row(df: pd.DataFrame) -> pd.DataFrame:
    df_columns = df[base_feature]
    df_prev_row_diff = df_columns.pct_change()
    df_prev_row_diff.columns = [name + "_diff" for name in df_columns.columns]
    df = df.join(df_prev_row_diff, how='left')
    return df


# Add columns that calculates the delta w.r.t a given row for each base feature
# This normalizes the data against given date
# Returns updated dataframe
def add_detla_from_date(df: pd.DataFrame, date: datetime) -> pd.DataFrame:
    df_columns = df[base_feature]
    row = df.index.get_loc(date)

    df_row0_diff = (df_columns - df_columns.iloc[row]) / df_columns.iloc[row]
    df_row0_diff.columns = [name + "_start" for name in df_columns.columns]
    df = df.join(df_row0_diff, how='left')
    return df


# Add earnings information to the dataframe
# Returns updated dataframe
def add_earnings_data(df: pd.DataFrame, ticker: Ticker, start_date: str,
                      end_date: str) -> pd.DataFrame:
    # Function to add earnings information
    earnings = ticker.earnings_dates
    earnings = earnings.sort_index()
    earnings = earnings.loc[start_date:end_date]
    earnings.index.name = "Earnings_Date"
    earnings.rename(columns={
        'EPS Estimate': 'EPS_Estimate',
        'Reported EPS': 'EPS_Reported',
        'Surprise(%)': 'Surprise(%)'
    },
                    inplace=True)

    # Join earnings with data frame
    df.index = df.index.normalize()
    earnings.index = earnings.index.normalize()
    df = df.join(earnings, how='left')
    df['Earnings_Date'] = df['EPS_Reported'].notna()
    return df


# Helper function to get an shited date
def get_date_back(date_str: str, delta_days: int) -> str:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    date_back = date_obj - timedelta(days=delta_days)
    return date_back.strftime("%Y-%m-%d")


def download_data(stock_symbol: str,
                  start_date: str,
                  end_date: str,
                  windows=[10, 20, 50]) -> pd.DataFrame:
    ticker = Ticker(stock_symbol)

    # We need to look back some time window so that all technical indicators are all valid.
    shifted_start_date = get_date_back(start_date, windows[-1] + 50)
    df = ticker.history(start=shifted_start_date, end=end_date, interval="1d")

    # Add technical indicator
    df = add_moving_averages(df, windows=windows)
    df = add_macd(df)

    # Trim data within the interested time window
    df = df.loc[start_date:end_date]
    # Add columns with normalized data
    df = add_delta_from_prev_row(df)
    df = add_detla_from_date(df, df.index[0])

    df = add_earnings_data(df, ticker, start_date, end_date)
    # Reformat the index to be just days
    df.index = df.index.date
    # Add a column for stock symbol
    df["stock"] = stock_symbol
    return df


def create_dataset_with_labels(stock_symbol: str,
                               start_date: str,
                               end_date: str,
                               vis: bool = False) -> pd.DataFrame:
    df = download_data(stock_symbol, start_date, end_date)
    if vis:
        visualize_dataset(df)

    # create labels and add them into the dataframe
    labels = compute_labels(df)
    df = df.join(labels, how='right')
    df = df.iloc[1:]  # drop 1st row
    return df


if __name__ == "__main__":
    stock = "AAPL"
    start_date, end_date = "2023-01-01", "2024-01-01"
    data = create_dataset_with_labels(stock, start_date, end_date)
    data.to_csv(f"./data/{stock}_{start_date}_{end_date}.csv", index=True)

    # print(data.shape[0])
    # print(data['MA_10'].count())
    # print(data['MA_20'].count())
    # print(data['MA_50'].count())
    # print(data['MACD_12_26_9'].count())
    # print(data['MACDh_12_26_9'].count())
    # print(data['MACDs_12_26_9'].count())
    # print(data["EPS_Estimate"].count())
    # print(data["Earnings_Date"].sum())
    #
    # print(f"5 days Up: {(data['trend_5days'] == 1).sum()}, 5 days Down: {(data['trend_5days'] == -1).sum()}")
    # print(f"10 days Up: {(data['trend_10days'] == 1).sum()}, 10 days Down: {(data['trend_10days'] == -1).sum()}")
    # print(f"30 days Up: {(data['trend_30days'] == 1).sum()}, 30 days Down: {(data['trend_30days'] == -1).sum()}")
