import pandas as pd


def add_bullish_bearish_signals(df: pd.DataFrame) -> pd.DataFrame:
    # shift df by 1 row so we can compare with prev day
    df_shift = df.shift(1)

    df = add_ma_crossover(df, df_shift)
    df = add_macd_crossover(df, df_shift)
    df = add_rsi_breakout(df)
    df = add_vwap_crossover(df, df_shift)
    df = add_bollinger_bands(df)
    df = add_price_over_ma(df)
    return df


def add_ma_crossover(df: pd.DataFrame, df_shift: pd.DataFrame) -> pd.DataFrame:
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

    return df


def add_macd_crossover(df: pd.DataFrame,
                       df_shift: pd.DataFrame) -> pd.DataFrame:
    # MACD Crossover: Detect when MACD line crosses above or below Signal line
    macd, signal = 'MACD_12_26_9', 'MACDs_12_26_9'
    df['MACD_Crossover_Signal'] = 0
    df.loc[(df[macd] > df[signal]) & (df_shift[macd] <= df_shift[signal]),
           'MACD_Crossover_Signal'] = 1  # Bullish MACD Crossover (Buy)
    df.loc[(df[macd] < df[signal]) & (df_shift[macd] >= df_shift[signal]),
           'MACD_Crossover_Signal'] = -1  # Bearish MACD Crossover (Sell)
    return df


def add_rsi_breakout(df: pd.DataFrame) -> pd.DataFrame:
    # RSI
    rsi = "RSI_14"
    df['RSI_Over_Bought_Signal'] = 0
    df.loc[(df[rsi] < 30),
           'RSI_Over_Bought_Signal'] = 1  # RSI is below 30 (oversold). (Buy)
    df.loc[
        (df[rsi] > 70),
        'RSI_Over_Bought_Signal'] = -1  # RSI is above 70 (overbought). (Sell)
    return df

    # # OBV Signal: Buy if OBV is increasing, Sell if decreasing
    # df['OBV_Signal'] = np.where(df['OBV'] > df['OBV'].shift(1), 1, -1)


def add_vwap_crossover(df: pd.DataFrame,
                       df_shift: pd.DataFrame) -> pd.DataFrame:
    # VWAP Signal: Buy if price is cross over VWAP, Sell if below
    df['VWAP_Crossover_Signal'] = 0
    df.loc[(df['Close'] > df['VWAP']) &
           (df_shift['Close'] <= df_shift['VWAP']),
           'VWAP_Crossover_Signal'] = 1  # Bullish VWAP Crossover (Buy)
    df.loc[(df['Close'] < df['VWAP']) &
           (df_shift['Close'] >= df_shift['VWAP']),
           'VWAP_Crossover_Signal'] = -1  # Bearish VWAP Crossover (Sell)
    return df


def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    # Bollinger Bands (BB) Signal:
    df['BB_Signal'] = 0
    df.loc[df['Close'] <= df['BB_Lower'],
           'BB_Signal'] = 1  # Buy when price hits lower band
    df.loc[df['Close'] >= df['BB_Upper'],
           'BB_Signal'] = -1  # Sell when price hits upper band
    return df

    # # ATR-Based Risk Management (Optional Stop-Loss Indicator)
    # df['ATR_Stop'] = df['Close'] - df['ATR']  # Example: Stop-loss at ATR below Close


def add_price_over_ma(df: pd.DataFrame) -> pd.DataFrame:
    # Price Above/Below 5day Moving Average
    df['Price_Above_MA_5'] = 0
    df.loc[df["Close"] > df["MA_5"], 'Price_Above_MA_5'] = 1
    df['Price_Below_MA_5'] = 0
    df.loc[df["Close"] < df["MA_5"], 'Price_Below_MA_5'] = 1

    return df
