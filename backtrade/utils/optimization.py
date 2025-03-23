"""
优化工具模块

这个模块包含了参数优化相关的函数。
"""

from backtrade.utils import create_bt_data_feed, load_data_from_yahoo

import backtrader as bt
import pandas as pd
import numpy as np
from itertools import product


def optimize_ma_strategy(data,
                         ma_short_range=(5, 20),
                         ma_long_range=(20, 100),
                         step=5,
                         commission=0.001,
                         initial_cash=100000):
    """
    针对移动平均策略进行参数优化
    
    参数:
    data: Backtrader的数据源或DataFrame
    ma_short_range: 短期均线的参数范围，如(5, 20)表示从5到20
    ma_long_range: 长期均线的参数范围，如(20, 100)表示从20到100
    step: 参数步长
    commission: 手续费率
    initial_cash: 初始资金
    
    返回:
    pandas.DataFrame: 包含不同参数组合及其回测结果的DataFrame
    """
    # 确保数据是backtrader feed
    if isinstance(data, pd.DataFrame):
        from .data_utils import df_to_btfeed
        data = df_to_btfeed(data)

    # 生成参数组合
    short_ma_values = list(
        range(ma_short_range[0], ma_short_range[1] + 1, step))
    long_ma_values = list(range(ma_long_range[0], ma_long_range[1] + 1, step))

    # 确保短期均线小于长期均线
    param_combinations = [
        (short, long)
        for short, long in product(short_ma_values, long_ma_values)
        if short < long
    ]

    # 存储每次优化的结果
    results = []

    # 遍历每个参数组合
    for short_ma, long_ma in param_combinations:
        # 创建cerebro实例
        cerebro = bt.Cerebro()
        cerebro.adddata(data)

        # 添加移动均线交叉策略
        class MAStrategy(bt.Strategy):
            params = dict(short_ma=short_ma, long_ma=long_ma)

            def __init__(self):
                self.short_ma = bt.indicators.SMA(self.data.close,
                                                  period=self.params.short_ma)
                self.long_ma = bt.indicators.SMA(self.data.close,
                                                 period=self.params.long_ma)
                self.crossover = bt.indicators.CrossOver(
                    self.short_ma, self.long_ma)

                # 交易记录
                self.trades = []
                self.buy_dates = []
                self.buy_prices = []
                self.sell_dates = []
                self.sell_prices = []

            def next(self):
                if self.crossover > 0:  # 金叉买入
                    self.buy()
                    self.buy_dates.append(self.data.datetime.date())
                    self.buy_prices.append(self.data.close[0])
                elif self.crossover < 0:  # 死叉卖出
                    self.sell()
                    self.sell_dates.append(self.data.datetime.date())
                    self.sell_prices.append(self.data.close[0])

            def stop(self):
                # 计算总收益率
                self.roi = (self.broker.getvalue() / initial_cash - 1.0) * 100

        # Add strategy with parameters
        strategy_params = {}
        cerebro.addstrategy(MAStrategy, **strategy_params)

        # 设置资金和手续费
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)

        # 运行回测
        cerebro.run()

        # 获取策略实例
        strategy = cerebro.runstrats[0][0]

        # 添加结果
        results.append({
            'short_ma': short_ma,
            'long_ma': long_ma,
            'roi': strategy.roi,
            'final_value': cerebro.broker.getvalue(),
            'n_trades': len(strategy.buy_dates)
        })

    # 转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roi',
                                        ascending=False).reset_index(drop=True)

    return results_df


def walk_forward_optimization(data,
                              strategy_class,
                              param_grid,
                              train_ratio=0.7,
                              initial_cash=100000,
                              commission=0.001):
    """
    进行移动窗口的步进优化（Walk Forward Optimization）
    
    参数:
    data: 回测数据源
    strategy_class: 策略类
    param_grid: 参数网格，格式为 {'param_name': [param_values]}
    train_ratio: 训练集比例
    initial_cash: 初始资金
    commission: 手续费率
    
    返回:
    tuple: (train_results, test_results, combined_results)
    """
    import backtrader as bt
    import pandas as pd
    import numpy as np
    from itertools import product

    # 确保数据是DataFrame格式
    if not isinstance(data, pd.DataFrame):
        raise ValueError("数据必须是pandas DataFrame格式")

    # 拆分为训练集和测试集
    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    print(f"训练集: {len(train_data)}行，测试集: {len(test_data)}行")

    # 生成参数组合
    param_names = list(param_grid.keys())
    param_values = list(product(*param_grid.values()))

    train_results = []
    test_results = []

    # 在训练集上优化参数
    from .data_utils import df_to_btfeed
    from .backtest_utils import run_backtest

    # 转换数据为backtrader feed
    train_feed = df_to_btfeed(train_data)
    test_feed = df_to_btfeed(test_data)

    # 在训练集上进行参数优化
    for params in param_values:
        param_dict = dict(zip(param_names, params))

        # 执行训练集回测
        train_result, _ = run_backtest(train_data, strategy_class, param_dict,
                                       initial_cash, commission)

        # 保存训练结果
        result_entry = {
            'params':
            param_dict,
            'total_return':
            train_result['total_return'],
            'sharpe_ratio':
            train_result['sharpe_ratio'],
            'max_drawdown':
            train_result['max_drawdown'],
            'total_trades':
            train_result['total_trades'],
            'win_rate':
            train_result['winning_trades'] /
            max(1, train_result['total_trades']) * 100
        }

        train_results.append(result_entry)

    # 按收益率排序训练结果
    train_results = sorted(train_results,
                           key=lambda x: x['total_return'],
                           reverse=True)

    # 获取最优参数
    best_params = train_results[0]['params']
    print(f"最优参数: {best_params}")

    # 在测试集上验证最优参数
    test_result, _ = run_backtest(test_data, strategy_class, best_params,
                                  initial_cash, commission)

    test_result_entry = {
        'params':
        best_params,
        'total_return':
        test_result['total_return'],
        'sharpe_ratio':
        test_result['sharpe_ratio'],
        'max_drawdown':
        test_result['max_drawdown'],
        'total_trades':
        test_result['total_trades'],
        'win_rate':
        test_result['winning_trades'] / max(1, test_result['total_trades']) *
        100
    }

    test_results.append(test_result_entry)

    # 合并结果
    combined_results = {
        'best_params':
        best_params,
        'train_performance':
        train_results[0],
        'test_performance':
        test_result_entry,
        'all_train_results':
        pd.DataFrame(train_results),
        'out_of_sample_ratio':
        test_result_entry['total_return'] /
        max(0.01, train_results[0]['total_return'])
    }

    return train_results, test_results, combined_results


def optimize_multi_objective(
        data,
        strategy_class,
        param_grid,
        objectives=['total_return', 'sharpe_ratio', 'max_drawdown'],
        weights=[0.5, 0.3, 0.2],
        initial_cash=100000,
        commission=0.001):
    """
    多目标优化函数，可以同时考虑多个指标
    
    参数:
    data: 回测数据
    strategy_class: 策略类
    param_grid: 参数网格，格式为 {'param_name': [param_values]}
    objectives: 优化目标列表，可选值包括 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'
    weights: 各个目标的权重
    initial_cash: 初始资金
    commission: 手续费率
    
    返回:
    pandas.DataFrame: 包含不同参数组合及其回测结果和综合得分的DataFrame
    """
    # 生成参数组合
    param_names = list(param_grid.keys())
    param_values = list(product(*param_grid.values()))

    from .backtest_utils import run_backtest
    from .data_utils import df_to_btfeed

    # 存储结果
    results = []

    # 对每个参数组合进行回测
    for params in param_values:
        param_dict = dict(zip(param_names, params))

        # 执行回测
        backtest_result, _ = run_backtest(data, strategy_class, param_dict,
                                          initial_cash, commission)

        # 提取指标
        metrics = {}
        for objective in objectives:
            if objective == 'win_rate':
                metrics[objective] = backtest_result['winning_trades'] / max(
                    1, backtest_result['total_trades']) * 100
            else:
                metrics[objective] = backtest_result.get(objective, 0)

        # 保存结果
        result_entry = {
            **param_dict,
            **metrics, 'total_trades': backtest_result['total_trades']
        }

        results.append(result_entry)

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 标准化指标（归一化到0-1之间）
    normalized_df = results_df.copy()

    for objective in objectives:
        if objective == 'max_drawdown':  # 最大回撤越小越好
            min_val = results_df[objective].min()
            max_val = results_df[objective].max()
            # 反转标准化，使较小的值得到更高的分数
            if max_val > min_val:
                normalized_df[f'norm_{objective}'] = 1 - (
                    results_df[objective] - min_val) / (max_val - min_val)
            else:
                normalized_df[f'norm_{objective}'] = 1.0
        else:  # 其他指标越大越好
            min_val = results_df[objective].min()
            max_val = results_df[objective].max()
            if max_val > min_val:
                normalized_df[f'norm_{objective}'] = (
                    results_df[objective] - min_val) / (max_val - min_val)
            else:
                normalized_df[f'norm_{objective}'] = 1.0

    # 计算综合得分
    normalized_df['score'] = 0
    for i, objective in enumerate(objectives):
        normalized_df[
            'score'] += weights[i] * normalized_df[f'norm_{objective}']

    # 按得分排序
    normalized_df = normalized_df.sort_values(
        'score', ascending=False).reset_index(drop=True)

    return normalized_df


if __name__ == "__main__":
    df = load_data_from_yahoo("TSLA", "2020-01-01", "2023-12-31", save_to_csv=False)
    bt_data = create_bt_data_feed(df)

    # 定义参数优化范围
    ma_short_range = (5, 20)   # 短期均线范围
    ma_long_range = (20, 50)   # 长期均线范围
    step = 5                   # 步长

    # 执行参数优化
    opt_results = optimize_ma_strategy(
        data=bt_data,
        ma_short_range=ma_short_range,
        ma_long_range=ma_long_range,
        step=step,
        commission=0.001,
        initial_cash=100000
    )

    # 显示优化结果
    opt_results.head(10)