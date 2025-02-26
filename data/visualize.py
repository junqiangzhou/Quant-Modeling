import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt


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
