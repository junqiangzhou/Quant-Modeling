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
