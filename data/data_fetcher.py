import pandas as pd
import yfinance as yf
from yahoo_fin import stock_info as si
import pandas_ta as ta
import numpy as np

from datetime import datetime, timedelta
from data.visualize import visualize_dataset
from data.label import create_labels

# List of basic data downloaded from Yahoo Finance
base_feature = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_20', 'MA_50'
]


def get_date_back(date_str: str, delta_days: int) -> str:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    date_back = date_obj - timedelta(days=delta_days)
    return date_back.strftime("%Y-%m-%d")


# Add columns that calculates the delta w.r.t. previous row for each base feature
# Returns updated dataframe
def add_delta_from_prev_row(df: pd.DataFrame) -> pd.DataFrame:
    df_columns = df[base_feature]
    df_prev_row_diff = df_columns.pct_change()
    df_prev_row_diff.columns = [name + "_diff" for name in df_columns.columns]

    df = df.join(df_prev_row_diff, how='left')
    return df


def add_row0_diff(df, date):
    df_columns = df[base_feature]
    row = df.index.get_loc(date)

    df_row0_diff = (df_columns - df_columns.iloc[row]) / df_columns.iloc[row]
    df_row0_diff.columns = [name + "_start" for name in df_columns.columns]

    df = df.join(df_row0_diff, how='left')
    return df




def download_data(stock_symbol: str, start_date: str,
                  end_date: str) -> pd.DataFrame:
    ticker = yf.Ticker(stock_symbol)

    # We need to look back some time window so that all technical indicators are all valid.
    shifted_start_date = get_date_back(start_date, 90)
    df = ticker.history(start=shifted_start_date, end=end_date, interval="1d")

    # Function to compute Moving Averages
    def add_moving_averages(df, windows=[10, 20, 50]):
        for window in windows:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        return df

    # Function to compute MACD
    def add_macd(df):
        macd = ta.macd(df['Close'])
        df = df.join(macd)
        return df

    # Add indicators to dataframe
    windows = [10, 20, 50]
    df = add_moving_averages(df, windows=windows)
    df = add_macd(df)
    df = add_delta_from_prev_row(df)
    df = add_row0_diff(df, df.index[windows[-1] + 1])

    # Function to add earnings information
    def create_earnings_data(ticker, start_date, end_date):
        earnings = ticker.earnings_dates
        earnings = earnings.sort_index()
        earnings = earnings.loc[start_date:end_date]
        # earnings = earnings[['EPS Estimate', 'Reported EPS']]
        # earnings.index = earnings.index.date
        earnings.index.name = "Earnings_Date"
        earnings.rename(columns={
            'EPS Estimate': 'EPS_Estimate',
            'Reported EPS': 'EPS_Reported',
            'Surprise(%)': 'Surprise(%)'
        },
                        inplace=True)

        return earnings

    def join_data(df, earnings):
        df.index = df.index.normalize()
        earnings.index = earnings.index.normalize()
        df = df.join(earnings, how='left')
        df['Earnings_Date'] = df['EPS_Reported'].notna()

        # latest time that has earnings info
        df = df.loc[start_date:end_date]
        return df

    earnings = create_earnings_data(ticker,
                                    start_date=start_date,
                                    end_date=end_date)
    df = join_data(df, earnings)
    return df


def create_dataset(stock_symbol: str,
                   start_date: str,
                   end_date: str,
                   vis: bool = False) -> pd.DataFrame:
    df = download_data(stock_symbol, start_date, end_date)
    if vis:
        visualize_dataset(df)
    labels = create_labels(df)
    data = df.join(labels, how='right')
    data.index = data.index.date
    data["stock"] = stock_symbol
    return data


if __name__ == "__main__":
    data = create_dataset("AAPL", "2023-01-01", "2024-01-01")

    print(data.shape[0])
    print(data['MA_10'].count())
    print(data['MA_20'].count())
    print(data['MA_50'].count())
    print(data['MACD_12_26_9'].count())
    print(data['MACDh_12_26_9'].count())
    print(data['MACDs_12_26_9'].count())
    print(data["EPS_Estimate"].count())
    print(data["Earnings_Date"].sum())
    #
    # print(f"5 days Up: {(data['trend_5days'] == 1).sum()}, 5 days Down: {(data['trend_5days'] == -1).sum()}")
    # print(f"10 days Up: {(data['trend_10days'] == 1).sum()}, 10 days Down: {(data['trend_10days'] == -1).sum()}")
    # print(f"30 days Up: {(data['trend_30days'] == 1).sum()}, 30 days Down: {(data['trend_30days'] == -1).sum()}")
