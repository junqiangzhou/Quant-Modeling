import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np


def visualize_dataset(df: pd.DataFrame) -> None:
    # Plot candlestick chart with indicators
    scale = (df["High"].max() + df["High"].min()) / 2

    label_lists = ["trend_30days+", "trend_30days-"]
    df_labels = df[label_lists].copy()
    for label in label_lists:
        df_labels[label] = df[label] * scale
        df_labels[label] = df[label].replace(0, np.nan)

    apds = [
        mpf.make_addplot(df[['MA_10', 'MA_20', 'MA_50']]),
        mpf.make_addplot(df_labels[label_lists[0]],
                         scatter=True,
                         marker="^",
                         color="green",
                         markersize=100),
        mpf.make_addplot(df_labels[label_lists[1]],
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
