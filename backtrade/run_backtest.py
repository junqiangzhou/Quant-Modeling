import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
import warnings

warnings.filterwarnings('ignore')

from backtrade.utils import df_to_btfeed, run_backtest, load_data_from_yahoo
from backtrade.strategy import VolumeBreakoutStrategy
from backtrade.utils import plot_performance_analysis, plot_backtest_results
from backtrade.utils import optimize_ma_strategy

plt.style.use("seaborn-darkgrid")
pd.set_option('display.max_columns', None)

df = load_data_from_yahoo("TSLA",
                          "2020-01-01",
                          "2023-12-31",
                          save_to_csv=False)

# 查看数据统计信息
print(f"数据时间范围: {df.index.min()} 至 {df.index.max()}")
print(f"共 {len(df)} 条记录")
print(f"数据列: {df.columns.tolist()}")

# 显示基本统计信息
df.describe()

# 执行回测
results, strategy = run_backtest(df=df,
                                 strategy_class=VolumeBreakoutStrategy,
                                 initial_cash=100000,
                                 commission=0.001)

fig = plot_backtest_results(df, results, max_candles=200)
fig.show()
fig, table = plot_performance_analysis(results)
fig.show()
table.show()

bt_data = df_to_btfeed(df)

# 定义参数优化范围
ma_short_range = (5, 20)  # 短期均线范围
ma_long_range = (20, 50)  # 长期均线范围
step = 5  # 步长

# 执行参数优化
opt_results = optimize_ma_strategy(data=bt_data,
                                   ma_short_range=ma_short_range,
                                   ma_long_range=ma_long_range,
                                   step=step,
                                   commission=0.001,
                                   initial_cash=100000)

# 显示优化结果
opt_results.head(10)
