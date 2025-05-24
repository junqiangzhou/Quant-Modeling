import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')

from data.stocks_fetcher import (MAG7, ETF, PICKS, BOND)
from backtrade.utils import run_multi_asset_backtest

from backtrade.multi_asset_strategy.ml_strategy import MultiAssetMLStrategy

from config.config import look_back_window
from data.data_fetcher import create_dataset
from data.utils import get_date_back
from feature.label import compute_labels

plt.style.use("seaborn-darkgrid")
pd.set_option('display.max_columns', None)

start_date = "2023-01-01"
end_date = "2024-12-31"
start_date_collect_data = get_date_back(start_date,
                                        (look_back_window + 50) * 1.5)

stocks = MAG7
df_all = None
for stock in stocks[:1]:
    print(f"\n>>>>>>>>>>stock: {stock}")
    try:
        df = create_dataset(stock, start_date_collect_data, end_date)
        start_date_backtest = get_date_back(start_date, look_back_window * 1.5)
        df = df[start_date_backtest:]
        df, _ = compute_labels(df)

        df_all = pd.concat([df_all, df],
                           ignore_index=False) if df_all is not None else df

    except:
        print(f"{stock} data not available")
        continue

strategy_class = MultiAssetMLStrategy
print(f"\n回测 {strategy_class.__name__}")

strategy_params = {
    'target_pct': 0.5,
    'debug_mode': True,
    'use_gt_label': True,
}
# Run with predicted labels
results, strategy = run_multi_asset_backtest(df_all=df_all,
                                             strategy_class=strategy_class,
                                             strategy_params=strategy_params,
                                             initial_cash=100000,
                                             commission=0.001)
