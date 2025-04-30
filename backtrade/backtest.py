import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from backtrade.utils import load_data_from_yahoo, run_backtest
from backtrade.utils import plot_performance_analysis, plot_backtest_results

from backtrade.strategy import (
    VolumeBreakoutStrategy,
    BollingerRSIStrategyV2,
    BollingerStrategyEnhanced,
    BuyAndHoldStrategy,
    DoubleMAStrategy,
    DMAStrategyIntradayImproved,
    DMABollPartialIntradayStrategy,
    MACrossoverStrategy,
    MLStrategy,
    NaiveRsiStrategy,
    TurtleStrategyImproved,
    MFIStrategy,
    OBVStrategy,
    RSIBBStrategy,
    VWAPStrategy,
)

from config.config import look_back_window
from data.data_fetcher import create_dataset
from data.utils import get_date_back

plt.style.use("seaborn-darkgrid")
pd.set_option('display.max_columns', None)

stock = "GOOG"
start_date = "2020-01-01"
end_date = "2024-12-31"
shifted_start_date = get_date_back(start_date, look_back_window + 30)
df = create_dataset(stock, shifted_start_date, end_date)
df.index = pd.to_datetime(df.index)
df = df[start_date:]

# # 查看数据统计信息
# print(f"数据时间范围: {df.index.min()} 至 {df.index.max()}")
# print(f"共 {len(df)} 条记录")
# print(f"数据列: {df.columns.tolist()}")

# # 显示基本统计信息
# df.describe()

# 执行回测
for strategy_class in [
        # VolumeBreakoutStrategy,
        ## BollingerRSIStrategyV2,
        # BollingerStrategyEnhanced,
        BuyAndHoldStrategy,
        # DoubleMAStrategy,
        # DMAStrategyIntradayImproved,
        ## DMABollPartialIntradayStrategy,
        # MACrossoverStrategy,
        MLStrategy,
        # NaiveRsiStrategy,
        # TurtleStrategyImproved,
        # MFIStrategy,
        # OBVStrategy,
        # RSIBBStrategy,
        # VWAPStrategy,
]:
    print(f"回测 {strategy_class.__name__}")
    results, strategy = run_backtest(df=df,
                                     strategy_class=strategy_class,
                                     initial_cash=100000,
                                     commission=0.001)

    viz = False
    if viz:
        fig = plot_backtest_results(df, results, max_candles=200)
        fig.show()
        fig, table = plot_performance_analysis(results)
        fig.show()
        table.show()

    print("\n")
