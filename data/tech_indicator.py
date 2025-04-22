import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import List

MA_WINDOWS = [5, 10, 20, 50]  # Default moving average windows


def add_tech_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', 'Close', and 'Volume' price columns.
        windows (list): List of window sizes for moving averages.
        Example: [10, 20, 50] for 10-day, 20-day, and 50-day moving averages.

    Returns:
        pd.DataFrame: DataFrame with technical indicators added.
    """
    # Add various technical indicators
    df = add_trading_volume(df)
    df = add_moving_averages(df)
    df = add_macd(df)
    df = add_kdj(df)
    df = add_rsi(df)
    df = add_obv(df)
    df = add_vwap(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)

    return df


def add_trading_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the trading volume and add it to the DataFrame.
    Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Volume' price columns.
    Returns:
        pd.DataFrame: DataFrame with trading volume added.
    """
    df['Trading_Volume'] = df['Volume'] * (df['High'] + df['Low']) / 2.0
    return df


def add_moving_averages(df: pd.DataFrame,
                        windows: List[int] = MA_WINDOWS) -> pd.DataFrame:
    """
    Compute moving averages and add them to the DataFrame.
    Parameters:
        df (pd.DataFrame): DataFrame containing 'Close' price column.
        windows (list): List of window sizes for moving averages.
    Returns:
        pd.DataFrame: DataFrame with moving averages added.
    """
    for window in windows:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the MACD indicator and add it to the DataFrame.
    Parameters:
        df (pd.DataFrame): DataFrame containing 'Close' price column.
    Returns:
        pd.DataFrame: DataFrame with MACD columns added.
    """
    macd = ta.macd(df['Close'])
    if macd is None:
        raise ValueError("MACD calculation failed.")
    df = df.join(macd)
    return df


def add_kdj(df: pd.DataFrame) -> pd.DataFrame:
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
    else:
        raise ValueError("KDJ calculation failed.")

    return df


def add_rsi(df: pd.DataFrame,
            column: str = 'Close',
            period: int = 14) -> pd.DataFrame:
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

    # Compute Relative Strength (RS)
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero

    # Compute RSI
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(method="bfill")

    df["RSI_14"] = rsi.values

    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the On-Balance Volume (OBV) for a given DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price data.
    """
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the Volume-Weighted Average Price (VWAP) for a given DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price data.
    """
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['Cumulative_Volume_Price'] = (df['Close'] * df['Volume']).cumsum()
    df['VWAP'] = df['Cumulative_Volume_Price'] / df['Cumulative_Volume']
    df.drop(columns=['Cumulative_Volume', 'Cumulative_Volume_Price'],
            inplace=True)  # Remove intermediate calculations

    return df


def add_bollinger_bands(df: pd.DataFrame,
                        rolling_window: int = 20,
                        num_std: int = 2) -> pd.DataFrame:
    """
    Add Bollinger Bands to a given DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price data.
    window (int): Lookback window for moving average.
    num_std (int): Number of standard deviations for upper and lower bands.
    """
    df['BB_Mid'] = df['Close'].rolling(
        window=rolling_window).mean()  # Middle Band
    df['BB_Std'] = df['Close'].rolling(window=rolling_window).std()

    if df['BB_Mid'].isna().all() or df['BB_Std'].isna().all():
        raise ValueError("NaN values found in Bollinger Bands calculation.")

    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * num_std)  # Upper Band
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * num_std)  # Lower Band

    return df


def add_atr(df: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """
    Add the Average True Range (ATR) for a given DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price data.
    atr_window (int): Lookback period for ATR calculation (default=14).
    """
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close'] = np.abs(df['High'] - df['Close'].shift(1))
    df['Low-Close'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-Close',
                   'Low-Close']].max(axis=1)  # True Range
    df['ATR'] = df['TR'].rolling(window=atr_window).mean()
    if df['ATR'].isna().all():
        raise ValueError("NaN values found in ATR calculation.")
    df.drop(columns=['High-Low', 'High-Close', 'Low-Close', 'TR'],
            inplace=True)  # Remove intermediate calculations
    return df
