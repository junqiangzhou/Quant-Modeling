import pandas as pd
import pandas_ta as ta
import numpy as np


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


def add_kdj(df):
    """
    Compute the KDJ indicator and add it to the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' price columns.

    Returns:
        pd.DataFrame: DataFrame with K, D, and J columns added.
    """
    # Compute Stochastic Oscillator (K, D)
    stoch = ta.stoch(df['High'], df['Low'], df['Close'])

    if stoch is not None:
        df = df.join(stoch)  # Adds 'STOCHk_14_3_3' (K) and 'STOCHd_14_3_3' (D)

        # Compute J Line: J = 3 * K - 2 * D
        df["J"] = 3 * df["STOCHk_14_3_3"] - 2 * df["STOCHd_14_3_3"]

    return df


def add_rsi(df, column='Close', period=14):
    """
    Add the Relative Strength Index (RSI) for a given DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price data.
    column (str): Column name for the closing price.
    period (int): Lookback period for RSI calculation (default=14).
    
    Returns:
    pd.Series: RSI values.
    """
    delta = df[column].diff(1)  # Calculate price changes

    # Separate gains and losses
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # Use exponential moving average (EMA) for stability
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period).mean()
    # print(avg_gain)
    # print(avg_loss)

    # Compute Relative Strength (RS)
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero

    # Compute RSI
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(method="bfill")

    df["RSI_14"] = rsi.values

    return df
