from data.utils import get_stock_df
from config.config import label_feature

import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_dataset(df: pd.DataFrame,
                      stock=None,
                      viz_label="trend_30days",
                      viz_pred=False,
                      plot_macd=True) -> None:
    assert viz_label in label_feature, "label not found in label_feature"
    if stock:
        df = get_stock_df(df, stock)
        if df.shape[0] == 0:
            print(f"No data for {stock}")
            return

    # Plot candlestick chart with indicators
    scale = (df["High"].max() + df["High"].min()) / 2

    df[viz_label + "+"] = np.where(df[viz_label] == 1, 1, np.nan) * scale
    df[viz_label + "-"] = np.where(df[viz_label] == 2, 1, np.nan) * scale

    apds = [mpf.make_addplot(df[['MA_10', 'MA_20', 'MA_50']])]
    # Plot GT label
    apds.append(
        mpf.make_addplot(df[viz_label + "+"],
                         scatter=True,
                         marker="^",
                         color="green",
                         markersize=100))
    apds.append(
        mpf.make_addplot(df[viz_label + "-"],
                         scatter=True,
                         marker=">",
                         color="red",
                         markersize=100))
    # Plot Predict label
    if viz_pred:
        pred_label = viz_label + "_pred"
        df[pred_label +
           "+"] = np.where(df[pred_label] == 1, 1, np.nan) * scale * 0.9
        df[pred_label +
           "-"] = np.where(df[pred_label] == 2, 1, np.nan) * scale * 0.9

        apds.append(
            mpf.make_addplot(df[pred_label + "+"],
                             scatter=True,
                             marker="^",
                             color="blue",
                             markersize=100))
        apds.append(
            mpf.make_addplot(df[pred_label + "-"],
                             scatter=True,
                             marker=">",
                             color="magenta",
                             markersize=100))

    mpf.plot(df,
             type='candle',
             volume=True,
             style='yahoo',
             addplot=apds,
             title=f'{stock} Daily Candlestick Chart with MA and MACD',
             figsize=(18, 10),
             ylabel='Price ($)',
             ylabel_lower='Volume',
             panel_ratios=(6, 2))

    # MACD plot
    if plot_macd:
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
    csv_file = "data/dataset/stock_testing_2023-01-01_2024-12-31.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Please run data_fetcher.py to download the data first.")
    else:
        df_all = pd.read_csv(csv_file)
        df_all['Date'] = pd.to_datetime(df_all['Date'])
        df_all.set_index('Date', inplace=True)

    stocks = df_all['stock'].unique()
    for stock in stocks:
        visualize_dataset(df_all,
                          stock=stock,
                          viz_label="trend_30days",
                          viz_pred=True,
                          plot_macd=False)
        plt.show()
