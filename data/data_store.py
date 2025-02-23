import pandas as pd
import yfinance as yf
from yahoo_fin import stock_info as si
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta

base_feature = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_20', 'MA_50'
]


def get_date_back(date_str: str, delta_days: int) -> str:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    date_back = date_obj - timedelta(days=delta_days)
    return date_back.strftime("%Y-%m-%d")


def add_prev_diff(df):
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


def visualize_dataset(df: pd.DataFrame) -> None:
    # Plot candlestick chart with indicators
    apds = [
        mpf.make_addplot(df[['MA_10', 'MA_20', 'MA_50']]),
    ]

    mpf.plot(df,
             type='candle',
             volume=True,
             style='yahoo',
             addplot=apds,
             title='AAPL Daily Candlestick Chart with MA and MACD (2023)',
             ylabel='Price ($)',
             ylabel_lower='Volume',
             panel_ratios=(6, 2))

    # MACD plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['MACD_12_26_9'], label='MACD', color='blue')
    ax.plot(df.index, df['MACDs_12_26_9'], label='Signal', color='orange')
    ax.bar(df.index,
           df['MACDh_12_26_9'],
           label='Hist',
           color='gray',
           alpha=0.5)
    ax.set_title('AAPL MACD (2023)')
    ax.set_ylabel('MACD')
    ax.legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()


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
    df = add_prev_diff(df)
    df = add_row0_diff(df, df.index[windows[-1] + 1])

    # Function to add earnings information
    def create_earnings_data(ticker, start_date, end_date):
        earnings = ticker.earnings_dates
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


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    up_threshold = 1 + 0.1
    down_threshold = 1 - 0.08

    next_earning_date_generator = (index for index, row in df.iterrows()
                                   if row["Earnings_Date"])
    curr_date = df.index[0]

    time_windows = [5, 10, 30]  # number of next rows to consider
    labels = pd.DataFrame(columns=[
        "trend_5days+", "trend_5days-", "trend_10days+", "trend_10days-",
        "trend_30days+", "trend_30days-"
    ])
    while next_date := next(next_earning_date_generator, None):
        # print(f"Quarter bewteen: {curr_date} and {next_date}")
        # somehow the earnings date is 1day ahead, so I need to start from index 2 here.
        df_window = df.loc[curr_date:next_date].iloc[:]

        for i in range(len(df_window)):
            # Get the index and row at position i
            index = df_window.index[i]
            row = df_window.iloc[i]
            curr_close = row["Close"]

            if i == 0 or i == len(df_window) - 1:
                labels.loc[index] = [
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                ]
                continue

            label = []
            for N in time_windows:
                # Calculate slice for next N rows, clamp to end
                end_pos = min(i + N, len(df_window))
                next_rows = df_window.iloc[i + 1:end_pos]
                # print(f"time window: start {index}, end {df_window.index[end_pos - 1]}")

                # Compute max and min of "Close"
                max_close, max_index = next_rows["Close"].max(
                ), next_rows["Close"].idxmax()
                min_close, min_index = next_rows["Close"].min(
                ), next_rows["Close"].idxmin()

                # print(f"Date: {index}, Close: {curr_close}, Max Date: {max_index}, Close: {max_close}, Min Date: {min_index}, Close: {min_close},")
                up, down = 0, 0
                if min_close > curr_close or max_close > curr_close * up_threshold and max_index < min_index:  # stright up
                    up = 1
                    # print(f"Date: {index}, Close: {row['Close']}, >>>>>Up: Percent {(max_close - curr_close) / curr_close * 100}, Length {max_index - index}")
                elif max_close < curr_close or min_close < curr_close * down_threshold and min_index < max_index:
                    down = 1
                    # print(f"Date: {index}, Close: {row['Close']}, <<<<<Down: Percent {(min_close - curr_close) / curr_close * 100}, Length {min_index - index}")
                label += [up, down]
            labels.loc[index] = label

        curr_date = next_date + timedelta(days=1)

    return labels


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
