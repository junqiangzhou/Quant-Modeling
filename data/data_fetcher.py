import pandas as pd
import numpy as np
import time
from yfinance import Ticker
import requests
# from yahoo_fin import stock_info as si

from datetime import datetime, timedelta
from data.indicator import (add_macd, add_moving_averages, add_kdj, add_rsi,
                            add_obv, add_vwap, add_bollinger_bands, add_atr,
                            add_buy_sell_signals, add_trading_volume,
                            add_bullish_bearish_pattern)
from data.stocks_fetcher import fetch_stocks
from data.label import one_hot_encoder
from config.config import base_feature


def get_stock_df(df_all: pd.DataFrame, stock: str) -> pd.DataFrame:
    df = df_all[df_all['stock'] == stock]
    return df


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


# Helper function to get an shited date
def get_date_back(date_str: str, delta_days: int) -> str:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    date_back = date_obj - timedelta(days=delta_days)
    return date_back.strftime("%Y-%m-%d")


def download_data(stock_symbol: str,
                  start_date: str,
                  end_date: str,
                  windows=[5, 10, 20, 50],
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
        print(
            f"stock {stock_symbol}, market cap: {int(market_cap / 1.0e9)}b"
        )

    # We need to look back some time window so that all technical indicators are all valid.
    shifted_start_date = get_date_back(start_date, windows[-1] + 50)
    end_date_inclusive = get_date_back(end_date, -1)
    df = ticker.history(start=shifted_start_date,
                        end=end_date_inclusive,
                        interval="1d")
    # Truncate to first 2 decimal digits (without rounding)
    df = df.applymap(lambda x: int(x * 100) / 100
                     if isinstance(x, float) else x)

    # Add technical indicator
    try:
        df = add_trading_volume(df)
        df = add_moving_averages(df, windows=windows)
        df = add_macd(df)
        df = add_kdj(df)
        df = add_rsi(df)
        df = add_obv(df)
        df = add_vwap(df)
        df = add_bollinger_bands(df)
        df = add_atr(df)
        df = add_buy_sell_signals(df)
        df = add_bullish_bearish_pattern(df)
    except Exception:
        raise ValueError(
            f"Technical indicators not available for {stock_symbol}")

    # Trim data within the interested time window
    df = df.loc[start_date:]
    # Add columns with normalized data
    df = add_delta_from_prev_row(df)
    df = add_daily_change(df)
    # df = add_detla_from_date(df, df.index[0])

    df = add_earnings_data(df, ticker, start_date, end_date_inclusive)
    # Add a column for stock symbol
    df["stock"] = stock_symbol
    return df


def create_dataset(stock_symbol: str,
                   start_date: str,
                   end_date: str,
                   session=None) -> pd.DataFrame:
    # Download raw data with technical indicators
    df = download_data(stock_symbol, start_date, end_date, session=session)
    if df is None:
        return None

    df = one_hot_encoder(df)
    # Reformat the index to be just days
    df.index = df.index.date
    return df


if __name__ == "__main__":
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    viz = False

    training_stocks, _ = fetch_stocks()
    print("# of stocks: ", len(training_stocks))
    # Generate training data
    print("Generate training data...")
    all_df = None
    session = requests.Session()
    for i, stock in enumerate(training_stocks):
        # print(">>>>>>stock: ", stock)
        if i >= 100 and i % 100 == 0:
            time.sleep(60)

        try:
            df = create_dataset(stock, start_date, end_date, session=session)
        except Exception as e:
            print(f"Error in processing {stock}: {e}")
            continue
        if df is None:
            continue

        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat([all_df, df], ignore_index=False)
    print("total # of training data points: ", all_df.shape[0])
    all_df.to_csv(f"./data/stock_training_{start_date}_{end_date}.csv",
                  index=True,
                  index_label="Date")
