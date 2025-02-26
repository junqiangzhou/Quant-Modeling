import pandas as pd
import pandas_ta as ta


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
