from datetime import datetime
import pandas as pd
import numpy as np
import requests
import time
import os
import bisect
from yfinance import Ticker

from data.tech_indicator import add_tech_indicators, MA_WINDOWS
from data.trend_indicator import add_bullish_bearish_signals
from data.stocks_fetcher import fetch_stocks
from feature.label import one_hot_encoder
from data.utils import get_date_back, get_stock_df
from config.config import base_feature


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


def add_daily_change(df: pd.DataFrame) -> pd.DataFrame:
    df["daily_change"] = (df["Close"] - df["Open"]) / df["Open"]
    return df


# Add earnings information to the dataframe
# Returns updated dataframe
def add_earnings_data(df: pd.DataFrame, ticker: Ticker, start_date: str,
                      end_date: str) -> pd.DataFrame:
    # Function to add earnings information
    earnings = ticker.earnings_dates
    if earnings is not None:
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
    else:
        df[["EPS_Estimate", "EPS_Reported", "Surprise(%)"]] = np.nan
        df['Earnings_Date'] = False
    return df


def download_data(stock_symbol: str,
                  start_date: str,
                  end_date: str,
                  session=None) -> pd.DataFrame:
    if session is None:
        ticker = Ticker(stock_symbol)
    else:
        ticker = Ticker(stock_symbol, session=session)

    if "marketCap" in ticker.info:
        market_cap = ticker.info["marketCap"]
        # eps = ticker.info["trailingEps"]
        # # Skip stocks with market cap less than 100 billion
        # if market_cap < 50.0e9 or eps < 0.0:  # 100 billion
        #     return None
        print(f"stock {stock_symbol}, market cap: {int(market_cap / 1.0e9)}b")

    # We need to look back some time window so that all technical indicators are all valid.
    shifted_start_date = get_date_back(start_date, MA_WINDOWS[-1] + 50)
    end_date_inclusive = get_date_back(end_date, -1)
    df = ticker.history(start=shifted_start_date,
                        end=end_date_inclusive,
                        interval="1d")

    df = add_earnings_data(df, ticker, start_date, end_date_inclusive)
    # Add a column for stock symbol
    df["stock"] = stock_symbol
    return df


def preprocess_data(df: pd.DataFrame, stock_symbol: str,
                    start_date: str) -> pd.DataFrame:
    # Truncate to first 2 decimal digits (without rounding)
    df = df.applymap(lambda x: int(x * 100) / 100
                     if isinstance(x, float) and pd.notnull(x) else x)

    # Add technical indicator
    try:
        df = add_tech_indicators(df)
        df = add_bullish_bearish_signals(df)
    except Exception:
        raise ValueError(
            f"Technical indicators not available for {stock_symbol}")

    # Trim data within the interested time window
    start_date = pd.to_datetime(start_date).tz_localize('UTC')  # convert string to datetime
    df = df.loc[start_date:]
    # Add columns with normalized data
    df = add_delta_from_prev_row(df)
    df = add_daily_change(df)
    # df = add_detla_from_date(df, df.index[0])

    df = one_hot_encoder(df)
    # Reformat the index to be just days
    df.index = df.index.date

    return df


def create_dataset(stock_symbol: str,
                   start_date: str,
                   end_date: str,
                   session=None) -> pd.DataFrame:
    # Download raw data with technical indicatorsd
    df = download_data(stock_symbol, start_date, end_date, session=session)
    if df is None:
        return None
    df = preprocess_data(df, stock_symbol, start_date)

    return df


def fetch_raw_training_dataset(stock_symbols: str,
                               start_date: str,
                               end_date: str,
                               csv_file: str,
                               session=None) -> pd.DataFrame:
    df_all = None
    for i, stock in enumerate(stock_symbols):
        if i >= 100 and i % 100 == 0:
            time.sleep(60)

        try:
            df = download_data(stock, start_date, end_date, session=session)
        except Exception:
            print(f"Error in processing {stock}: {e}")
            continue
        if df is None:
            continue

        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=False)
    print("total # of training data points: ", df_all.shape[0])
    df_all.to_csv(csv_file, index=True, index_label="Date")
    return


def fetch_training_dataset(df_raw_data: pd.DataFrame, stock_symbols: str,
                           start_date: str, end_date: str) -> pd.DataFrame:
    df_all = None
    for i, stock in enumerate(stock_symbols):
        try:
            df = get_stock_df(df_raw_data, stock)
        except Exception as e:
            continue
        if df is None:
            continue

        df = preprocess_data(df, stock, start_date)

        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=False)

    return df_all


if __name__ == "__main__":
    start_date = "2023-04-01"
    end_date = "2025-03-31"
    viz = False

    training_stocks, _ = fetch_stocks()
    print("# of stocks: ", len(training_stocks))
    # Generate training data
    print("Generate training data...")
    session = requests.Session()

    csv_file = f"./data/dataset/stock_training_{start_date}_{end_date}_raw_data.csv"
    if not os.path.exists(csv_file):
        fetch_raw_training_dataset(stock_symbols=training_stocks,
                                   start_date=start_date,
                                   end_date=end_date,
                                   csv_file=csv_file,
                                   session=None)

    df_raw_data = pd.read_csv(csv_file)
    df_raw_data.set_index('Date', inplace=True)
    df_raw_data.index = pd.to_datetime(df_raw_data.index, utc=True)
    df_all = fetch_training_dataset(df_raw_data=df_raw_data,
                                    stock_symbols=training_stocks,
                                    start_date=start_date,
                                    end_date=end_date)
    print("total # of training data points: ", df_all.shape[0])
    df_all.to_csv(f"./data/dataset/stock_training_{start_date}_{end_date}.csv",
                  index=True,
                  index_label="Date")
