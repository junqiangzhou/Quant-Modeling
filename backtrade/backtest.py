import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')

from data.stocks_fetcher import (MAG7, ETF, PICKS, BOND)
from backtrade.utils import load_data_from_yahoo, run_backtest
from backtrade.utils import plot_performance_analysis, plot_backtest_results

from backtrade.strategy import (
    # VolumeBreakoutStrategy,
    # BollingerRSIStrategyV2,
    # BollingerStrategyEnhanced,
    # DoubleMAStrategy,
    # DMAStrategyIntradayImproved,
    # DMABollPartialIntradayStrategy,
    # MACrossoverStrategy,
    # NaiveRsiStrategy,
    # TurtleStrategyImproved,
    # MFIStrategy,
    # OBVStrategy,
    # RSIBBStrategy,
    # VWAPStrategy,
    BuyAndHoldStrategy,
    MLStrategy,
)

from config.config import look_back_window
from data.data_fetcher import create_dataset
from data.stocks_fetcher import fetch_stocks
from data.utils import get_date_back
from feature.label import compute_labels

plt.style.use("seaborn-darkgrid")
pd.set_option('display.max_columns', None)

start_date = "2023-01-01"
end_date = "2024-12-31"
shifted_start_date = get_date_back(start_date, look_back_window + 30)

strategy_names = ["B&H", "ML"]
trade_metrics = [
    "total_return", "sharpe_ratio", "max_drawdown", "total_trades", "win_rate"
]

all_trade_metrics = [
    f"{metric}_{strategy}" for strategy in strategy_names
    for metric in trade_metrics
]
metrics_df = pd.DataFrame(columns=all_trade_metrics)

# 执行回测
# VolumeBreakoutStrategy,
## BollingerRSIStrategyV2,
# BollingerStrategyEnhanced,
# DoubleMAStrategy,
# DMAStrategyIntradayImproved,
## DMABollPartialIntradayStrategy,
# MACrossoverStrategy,
# NaiveRsiStrategy,
# TurtleStrategyImproved,
# MFIStrategy,
# OBVStrategy,
# RSIBBStrategy,
# VWAPStrategy,

stocks = MAG7 + PICKS
for stock in stocks:
    print(f"\n>>>>>>>>>>stock: {stock}")
    try:
        df = create_dataset(stock, shifted_start_date, end_date)
        df, _ = compute_labels(df)
        df.index = pd.to_datetime(df.index)
        df = df[start_date:]
    except:
        print(f"{stock} data not available")
        continue

    # # 查看数据统计信息
    # print(f"数据时间范围: {df.index.min()} 至 {df.index.max()}")
    # print(f"共 {len(df)} 条记录")
    # print(f"数据列: {df.columns.tolist()}")

    # # 显示基本统计信息
    # df.describe()

    metrics = []
    for strategy_class in [BuyAndHoldStrategy, MLStrategy]:
        print(f"\n回测 {strategy_class.__name__}")

        daily_change_perc = np.percentile(np.abs(df["daily_change"]), 99)
        if strategy_class == MLStrategy:
            strategy_params = {
                'target_pct': 0.9,
                'daily_change_perc': daily_change_perc,
                'debug_mode': False,
                'use_gt_label': False,
            }
            results, strategy = run_backtest(df=df,
                                             strategy_class=strategy_class,
                                             strategy_params=strategy_params,
                                             initial_cash=100000,
                                             commission=0.001)
        else:
            results, strategy = run_backtest(df=df,
                                             strategy_class=strategy_class,
                                             initial_cash=100000,
                                             commission=0.001)

        # viz = False
        # if viz:
        #     fig = plot_backtest_results(df, results, max_candles=200)
        #     fig.show()
        #     fig, table = plot_performance_analysis(results)
        #     fig.show()
        #     table.show()

        metrics += [results[metric] for metric in trade_metrics]
    metrics_df.loc[stock] = metrics

metrics_df = metrics_df.round(2)
metrics_df.to_csv(
    f"./backtrade/backtest_results/backtrade_tests_{start_date}_{end_date}.csv",
    index=True,
    index_label="Stock")
