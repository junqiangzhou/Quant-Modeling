import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import os
from data.data_fetcher import get_stock_df


def visualize_dataset(df: pd.DataFrame,
                      stock=None,
                      viz_labels=["trend_30days+", "trend_30days-"]) -> None:
    assert len(
        viz_labels) == 2, "Only 2 labels are supported for visualization"
    if stock:
        df = get_stock_df(df, stock)
    print(df.shape)

    # Plot candlestick chart with indicators
    scale = (df["High"].max() + df["High"].min()) / 2

    df_labels = df[viz_labels].copy()
    for label in viz_labels:
        df_labels[label] = df[label] * scale
        df_labels[label] = df[label].replace(0, np.nan)

    apds = [
        mpf.make_addplot(df[['MA_10', 'MA_20', 'MA_50']]),
        mpf.make_addplot(df_labels[viz_labels[0]],
                         scatter=True,
                         marker="^",
                         color="green",
                         markersize=100),
        mpf.make_addplot(df_labels[viz_labels[1]],
                         scatter=True,
                         marker=">",
                         color="red",
                         markersize=100)
    ]
    stock = df["stock"].iloc[0]

    mpf.plot(df,
             type='candle',
             volume=True,
             style='yahoo',
             addplot=apds,
             title=f'{stock} Daily Candlestick Chart with MA and MACD',
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
    ax.set_title(f'{stock} MACD')
    ax.set_ylabel('MACD')
    ax.legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    csv_file = "data/stock_training_2023-01-01_2024-12-31.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Please run data_fetcher.py to download the data first.")
    else:
        df_all = pd.read_csv(csv_file)
    print(df_all.index[0])
    df_all.index = pd.to_datetime(df_all.index)
    visualize_dataset(df_all, stock="GOOGL")
    plt.show()
