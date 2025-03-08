import pandas as pd
import pandas_ta as ta
import numpy as np


# Function to compute trading volume:
def add_trading_volume(df):
    df['Trading_Volume'] = df['Volume'] * (df['High'] + df['Low']) / 2.0
    return df


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


def add_obv(df):
    """
    Add the On-Balance Volume (OBV) for a given DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price data.
    """
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df


def add_vwap(df):
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


def add_bollinger_bands(df, rolling_window=20, num_std=2):
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
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * num_std)  # Upper Band
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * num_std)  # Lower Band

    return df


def add_atr(df, atr_window=14):
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
    df.drop(columns=['High-Low', 'High-Close', 'Low-Close', 'TR'],
            inplace=True)  # Remove intermediate calculations
    return df


def add_buy_sell_signals(df):
    # shift df by 1 row so we can compare with prev day
    df_shift = df.shift(1)

    # Moving Average Crossover: Detect when short_time MA crosses above or below long_time MA
    df['MA_5_10_Crossover_Signal'] = 0
    short_time, long_time = "MA_5", "MA_10"
    df.loc[(df[short_time] > df[long_time]) &
           (df_shift[short_time] <= df_shift[long_time]),
           'MA_5_10_Crossover_Signal'] = 1  # Golden Cross (Buy)
    df.loc[(df[short_time] < df[long_time]) &
           (df_shift[short_time] >= df_shift[long_time]),
           'MA_5_10_Crossover_Signal'] = -1  # Death Cross (Sell)

    df['MA_5_20_Crossover_Signal'] = 0
    short_time, long_time = "MA_5", "MA_20"
    df.loc[(df[short_time] > df[long_time]) &
           (df_shift[short_time] <= df_shift[long_time]),
           'MA_5_20_Crossover_Signal'] = 1  # Golden Cross (Buy)
    df.loc[(df[short_time] < df[long_time]) &
           (df_shift[short_time] >= df_shift[long_time]),
           'MA_5_20_Crossover_Signal'] = -1  # Death Cross (Sell)

    df['MA_5_50_Crossover_Signal'] = 0
    short_time, long_time = "MA_5", "MA_50"
    df.loc[(df[short_time] > df[long_time]) &
           (df_shift[short_time] <= df_shift[long_time]),
           'MA_5_50_Crossover_Signal'] = 1  # Golden Cross (Buy)
    df.loc[(df[short_time] < df[long_time]) &
           (df_shift[short_time] >= df_shift[long_time]),
           'MA_5_50_Crossover_Signal'] = -1  # Death Cross (Sell)

    df['MA_10_20_Crossover_Signal'] = 0
    short_time, long_time = "MA_10", "MA_20"
    df.loc[(df[short_time] > df[long_time]) &
           (df_shift[short_time] <= df_shift[long_time]),
           'MA_10_20_Crossover_Signal'] = 1  # Golden Cross (Buy)
    df.loc[(df[short_time] < df[long_time]) &
           (df_shift[short_time] >= df_shift[long_time]),
           'MA_10_20_Crossover_Signal'] = -1  # Death Cross (Sell)

    df['MA_10_50_Crossover_Signal'] = 0
    short_time, long_time = "MA_10", "MA_50"
    df.loc[(df[short_time] > df[long_time]) &
           (df_shift[short_time] <= df_shift[long_time]),
           'MA_10_50_Crossover_Signal'] = 1  # Golden Cross (Buy)
    df.loc[(df[short_time] < df[long_time]) &
           (df_shift[short_time] >= df_shift[long_time]),
           'MA_10_50_Crossover_Signal'] = -1  # Death Cross (Sell)

    df['MA_20_50_Crossover_Signal'] = 0
    short_time, long_time = "MA_20", "MA_50"
    df.loc[(df[short_time] > df[long_time]) &
           (df_shift[short_time] <= df_shift[long_time]),
           'MA_20_50_Crossover_Signal'] = 1  # Golden Cross (Buy)
    df.loc[(df[short_time] < df[long_time]) &
           (df_shift[short_time] >= df_shift[long_time]),
           'MA_20_50_Crossover_Signal'] = -1  # Death Cross (Sell)

    # MACD Crossover: Detect when MACD line crosses above or below Signal line
    macd, signal = 'MACD_12_26_9', 'MACDs_12_26_9'
    df['MACD_Crossover_Signal'] = 0
    df.loc[(df[macd] > df[signal]) & (df_shift[macd] <= df_shift[signal]),
           'MACD_Crossover_Signal'] = 1  # Bullish MACD Crossover (Buy)
    df.loc[(df[macd] < df[signal]) & (df_shift[macd] >= df_shift[signal]),
           'MACD_Crossover_Signal'] = -1  # Bearish MACD Crossover (Sell)

    # RSI
    rsi = "RSI_14"
    df['RSI_Over_Bought_Signal'] = 0
    df.loc[(df[rsi] < 30),
           'RSI_Over_Bought_Signal'] = 1  # RSI is below 30 (oversold). (Buy)
    df.loc[
        (df[rsi] > 70),
        'RSI_Over_Bought_Signal'] = -1  # RSI is above 70 (overbought). (Sell)

    # # OBV Signal: Buy if OBV is increasing, Sell if decreasing
    # df['OBV_Signal'] = np.where(df['OBV'] > df['OBV'].shift(1), 1, -1)

    # VWAP Signal: Buy if price is cross over VWAP, Sell if below
    df['VWAP_Crossover_Signal'] = 0
    df.loc[(df['Close'] > df['VWAP']) &
           (df_shift['Close'] <= df_shift['VWAP']),
           'VWAP_Crossover_Signal'] = 1  # Bullish VWAP Crossover (Buy)
    df.loc[(df['Close'] < df['VWAP']) &
           (df_shift['Close'] >= df_shift['VWAP']),
           'VWAP_Crossover_Signal'] = -1  # Bearish VWAP Crossover (Sell)

    # Bollinger Bands (BB) Signal:
    df['BB_Signal'] = 0
    df.loc[df['Close'] <= df['BB_Lower'],
           'BB_Signal'] = 1  # Buy when price hits lower band
    df.loc[df['Close'] >= df['BB_Upper'],
           'BB_Signal'] = -1  # Sell when price hits upper band

    # # ATR-Based Risk Management (Optional Stop-Loss Indicator)
    # df['ATR_Stop'] = df['Close'] - df['ATR']  # Example: Stop-loss at ATR below Close

    return df
