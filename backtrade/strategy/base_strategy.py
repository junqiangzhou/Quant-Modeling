"""
策略工具模块

这个模块包含了不同类型的交易策略实现。
"""

import backtrader as bt
import pandas as pd


class BaseStrategy(bt.Strategy):
    """
    通用策略基类，集成信号收集、日志记录和资产净值记录功能
    """
    # 日志等级
    LOG_LEVEL_DEBUG = 0
    LOG_LEVEL_INFO = 1
    LOG_LEVEL_WARNING = 2
    LOG_LEVEL_ERROR = 3

    params = (
        ('log_level', LOG_LEVEL_INFO),  # 日志级别
        ('collect_signals', True),  # 是否收集交易信号
    )

    def __init__(self):
        # 初始化交易信号列表
        self.buy_signals = []  # 买入信号列表，格式为 (datetime, price)
        self.sell_signals = []  # 卖出信号列表，格式为 (datetime, price)
        self.position_size = []  # 持仓变化列表，格式为 (datetime, size)
        self.logs = []  # 日志列表，格式为 (datetime, log_level, message)

        # 跟踪持仓和买入价格
        self.bar_executed = None
        self.buy_price = None
        self.position_value = 0  # 记录持仓数量

        # 新增：记录资产净值曲线，存放 (datetime, broker.getvalue())
        self.equity_curve = []

    def log(self, txt, dt=None, level=None):
        """记录日志"""
        if level is None:
            level = self.params.log_level

        if level >= self.params.log_level:
            dt = dt or self.datas[0].datetime.date(0)
            self.logs.append((dt, level, txt))
            print(f'{dt.isoformat()}: {txt}')

    def notify_order(self, order):
        """订单状态更新通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'买入执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}'
                )
                if self.params.collect_signals:
                    self.buy_signals.append(
                        (self.datas[0].datetime.datetime(0),
                         order.executed.price))
                # 累加持仓数量
                self.position_value += order.executed.size
                self.position_size.append(
                    (self.datas[0].datetime.datetime(0), self.position_value))
            elif order.issell():
                self.log(
                    f'卖出执行: 价格={order.executed.price:.2f}, 数量={abs(order.executed.size)}, 收入={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}'
                )
                if self.params.collect_signals:
                    self.sell_signals.append(
                        (self.datas[0].datetime.datetime(0),
                         order.executed.price))
                # 减少持仓数量
                self.position_value -= abs(order.executed.size)
                self.position_size.append(
                    (self.datas[0].datetime.datetime(0), self.position_value))
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单被拒绝或取消: {order.status}', level=self.LOG_LEVEL_WARNING)

    def notify_trade(self, trade):
        """交易完成通知"""
        if trade.isclosed:
            self.log(f'交易利润: 毛利={trade.pnl:.2f}, 净利={trade.pnlcomm:.2f}')

    def calc_max_shares(self, price):
        """计算在当前价格下能够购买的最大股票数量（考虑手续费）"""
        cash = self.broker.getcash()
        commission_rate = self.broker.getcommissioninfo(self.data).p.commission

        # cash = shares * price * (1 + commission_rate)
        max_shares = int(cash / (price * (1 + commission_rate)))
        return max_shares

    def next(self):
        """
        策略核心逻辑，每个Bar调用。
        请注意：如果子类覆盖 next() 方法，请调用 super().next() 以确保资产净值记录正常。
        """
        # 记录当前Bar的时间和资产净值
        dt = self.datas[0].datetime.datetime(0)
        self.equity_curve.append((dt, self.broker.getvalue()))

        # 子类实现具体逻辑
        pass

    def stop(self):
        """策略结束时调用"""
        self.log(f'策略结束: 最终资金={self.broker.getvalue():.2f}')

    def get_signals(self):
        """获取所有交易信号"""
        return {
            'buy': self.buy_signals,
            'sell': self.sell_signals,
            'position_size': self.position_size
        }

    def get_logs(self):
        """获取所有日志"""
        return self.logs

    def get_equity_curve(self):
        """
        返回资产净值曲线，格式为 pandas Series，其中 index 为时间，值为资产净值。
        """
        if not self.equity_curve:
            return pd.Series()
        times, values = zip(*self.equity_curve)
        return pd.Series(data=values, index=pd.to_datetime(times))


class CustomStrategy(BaseStrategy):
    params = (
        ('ma_period', 20),  # 移动平均周期  
        ('rsi_period', 14),  # RSI周期  
        ('rsi_overbought', 70),  # RSI超买水平  
        ('rsi_oversold', 30),  # RSI超卖水平  
        # 继承BaseStrategy的参数
        ('log_level', BaseStrategy.LOG_LEVEL_INFO),
        ('collect_signals', True),
    )

    def __init__(self):
        # 调用父类初始化
        BaseStrategy.__init__(self)

        # 添加指标
        self.ma = bt.indicators.SMA(self.data.close,
                                    period=self.params.ma_period)
        self.rsi = bt.indicators.RSI(self.data.close,
                                     period=self.params.rsi_period)

    def next(self):
        # 如果没有持仓
        if not self.position:
            # 当价格在MA之上且RSI超卖时买入
            if self.data.close[0] > self.ma[0] and self.rsi[
                    0] < self.params.rsi_oversold:
                # 计算可购买的最大股数
                max_shares = self.calc_max_shares(self.data.close[0])
                if max_shares > 0:
                    self.log(
                        f'买入信号: 价格={self.data.close[0]:.2f}, 数量={max_shares}, RSI={self.rsi[0]:.2f}'
                    )
                    self.buy(size=max_shares)
                    self.bar_executed = len(self)
                    self.buy_price = self.data.close[0]

        # 如果有持仓
        else:
            current_position_size = self.position.size

            # 当价格在MA之下或RSI超买时卖出
            if self.data.close[0] < self.ma[0] or self.rsi[
                    0] > self.params.rsi_overbought:
                self.log(
                    f'卖出信号: 价格={self.data.close[0]:.2f}, 持仓数量={current_position_size}, RSI={self.rsi[0]:.2f}'
                )
                self.close()
                return

            # 止损: 亏损超过5%
            if self.data.close[0] < self.buy_price * 0.95:
                self.log(
                    f'止损卖出: 价格={self.data.close[0]:.2f}, 持仓数量={current_position_size}'
                )
                self.close()
                return
