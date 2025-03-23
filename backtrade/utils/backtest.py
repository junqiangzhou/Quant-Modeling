######################################
# 回测工具模块 (backtest_tool.py)
######################################
import backtrader as bt
import pandas as pd
import numpy as np


def run_backtest(df,
                 strategy_class,
                 strategy_params=None,
                 initial_cash=100000.0,
                 commission=0.001):
    """
    运行回测，返回回测结果字典和策略实例
    
    参数:
      df: 包含OHLCV数据的DataFrame
      strategy_class: 回测使用的策略类（假设已实现 get_signals() 方法）
      strategy_params: 策略参数字典
      initial_cash: 初始资金
      commission: 交易佣金比例
    """
    cerebro = bt.Cerebro()

    # 添加策略
    if strategy_params:
        cerebro.addstrategy(strategy_class, **strategy_params)
    else:
        cerebro.addstrategy(strategy_class)

    # 确保 DataFrame 索引为 DatetimeIndex 或包含 trade_time 列
    if not isinstance(df.index,
                      pd.DatetimeIndex) and 'trade_time' in df.columns:
        df = df.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df.set_index('trade_time', inplace=True)

    # 定义一个自定义的 PandasData 类
    class PandasDataCustom(bt.feeds.PandasData):
        params = (
            ('datetime', None),  # 使用索引作为日期
            ('open', 'Open'),
            ('high', 'High'),
            ('low', 'Low'),
            ('close', 'Close'),
            ('volume', 'Volume'),
            ('openinterest', None))

    data = PandasDataCustom(dataname=df)
    cerebro.adddata(data)

    # 设置初始资金和佣金
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')

    print(f'初始资金: {initial_cash:.2f}')
    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100

    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get(
        'sharperatio', 0.0)
    if not isinstance(sharpe_ratio, float) or np.isnan(sharpe_ratio):
        sharpe_ratio = 0.0

    max_drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get(
        'drawdown', 0.0)

    trade_analysis = strat.analyzers.trade.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    winning_trades = trade_analysis.get('won', {}).get('total', 0)
    losing_trades = trade_analysis.get('lost', {}).get('total', 0)
    win_rate = (winning_trades / total_trades *
                100) if total_trades > 0 else 0.0

    print(f'最终资金: {final_value:.2f}')
    print(f'总收益率: {total_return:.2f}%')
    print(f'夏普比率: {sharpe_ratio:.2f}')
    print(f'最大回撤: {max_drawdown:.2f}%')
    print(f'总交易次数: {total_trades}')
    print(f'胜率: {win_rate:.2f}%')

    # 获取并标准化交易信号（假设策略中实现了 get_signals() 方法）
    signals = strat.get_signals()
    if signals is None:
        signals = {}
    signals.setdefault('buy', [])
    signals.setdefault('sell', [])

    # 如果信号元素为字典，则提取其中的 'time' 字段
    if signals['buy'] and isinstance(signals['buy'][0], dict):
        signals['buy'] = [sig.get('time') for sig in signals['buy']]
    if signals['sell'] and isinstance(signals['sell'][0], dict):
        signals['sell'] = [sig.get('time') for sig in signals['sell']]

    return {
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'signals':
        signals  # 格式：{'buy': [time1, time2, ...], 'sell': [time1, time2, ...]}
    }, strat
