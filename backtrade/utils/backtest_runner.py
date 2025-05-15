######################################
# 回测工具模块 (backtest_tool.py)
######################################
import backtrader as bt
import pandas as pd
import numpy as np

from config.config import (feature_names, buy_sell_signals, label_names)


class BacktestRunner:
    """
    回测执行器
    """

    def __init__(self,
                 strategy_class,
                 data_feed,
                 cash=100000.0,
                 commission=0.001,
                 **strategy_params):
        self.strategy_class = strategy_class
        self.data_feed = data_feed
        self.cash = cash
        self.commission = commission
        self.strategy_params = strategy_params

        # 初始化cerebro引擎
        self.cerebro = bt.Cerebro(cheat_on_close=True)
        self.cerebro.addstrategy(self.strategy_class, **self.strategy_params)

        # 添加数据
        self.cerebro.adddata(self.data_feed)

        # 设置初始资金和手续费
        self.cerebro.broker.setcash(self.cash)
        self.cerebro.broker.setcommission(commission=self.commission)

        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    def run(self, plot=True, figsize=(15, 10)):
        """
        运行回测
        
        参数:
        plot (bool): 是否绘制结果图表
        figsize (tuple): 图表大小
        
        返回:
        dict: 回测结果
        """
        # 运行回测
        results = self.cerebro.run()
        strat = results[0]

        # 获取回测结果
        initial_value = self.cash
        final_value = self.cerebro.broker.getvalue()

        # 计算回报
        total_return = (final_value - initial_value) / initial_value

        # 获取分析器结果
        sharpe_ratio = strat.analyzers.sharpe.get_analysis().get(
            'sharperatio', 0)
        if isinstance(sharpe_ratio, dict):
            sharpe_ratio = sharpe_ratio.get('sharperatio', 0)

        max_drawdown = strat.analyzers.drawdown.get_analysis().get(
            'max', {}).get('drawdown', 0)

        # 打印结果
        print(f"初始资金: {initial_value:.2f}")
        print(f"最终资金: {final_value:.2f}")
        print(f"总收益率: {total_return:.2%}")
        print(f"夏普比率: {sharpe_ratio:.4f}")
        print(f"最大回撤: {max_drawdown:.2%}")

        # 获取交易分析
        trade_analysis = strat.analyzers.trades.get_analysis()

        # 获取胜率
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        win_rate = won_trades / total_trades if total_trades > 0 else 0

        # 获取平均收益
        won_pnl = trade_analysis.get('won', {}).get('pnl', 0)
        lost_pnl = trade_analysis.get('lost', {}).get('pnl', 0)
        avg_won = won_pnl / won_trades if won_trades > 0 else 0
        avg_lost = lost_pnl / (total_trades - won_trades) if (
            total_trades - won_trades) > 0 else 0

        print(f"交易次数: {total_trades}")
        print(f"胜率: {win_rate:.2%}")
        print(f"平均盈利: {avg_won:.2f}")
        print(f"平均亏损: {avg_lost:.2f}")

        # 绘制结果
        if plot:
            self.cerebro.plot(style='candle', figsize=figsize)

        # 返回结果汇总
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_won': avg_won,
            'avg_lost': avg_lost,
            'strategy': strat
        }


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
        lines = tuple(feature_names + buy_sell_signals + label_names +
                      ["Earnings_Date"])
        params = (
            ('datetime', None),  # 使用索引作为日期
            ('open', 'Open'),
            ('high', 'High'),
            ('low', 'Low'),
            ('close', 'Close'),
            ('volume', 'Volume'),
            ('openinterest', None))
        params = params + tuple((feature, -1) for feature in lines)

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
    signals = {}
    if hasattr(strat, 'get_signals'):
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
