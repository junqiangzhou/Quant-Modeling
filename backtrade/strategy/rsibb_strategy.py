import backtrader as bt
from .base_strategy import BaseStrategy


class RSIBBStrategy(BaseStrategy):
    """
    基于RSI和布林带的交易策略

    策略逻辑：
      只有当RSI和布林带两个指标同时给出相同方向的信号时，才执行买入或卖出操作：
        - 买入条件：RSI从超卖区域向上突破（当前RSI > rsi_oversold且前一BarRSI <= rsi_oversold）
                      且价格从下方突破布林带下轨（当前收盘价 > 下轨且前一Bar收盘价 <= 下轨）。
        - 卖出条件：RSI从超买区域向下突破（当前RSI < rsi_overbought且前一BarRSI >= rsi_overbought）
                      且价格从上方跌破布林带上轨（当前收盘价 < 上轨且前一Bar收盘价 >= 上轨）。

    参数：
      - rsi_period: RSI指标计算周期，默认14
      - rsi_oversold: RSI超卖阈值，默认30
      - rsi_overbought: RSI超买阈值，默认70
      - bb_period: 布林带计算周期，默认20
      - bb_dev: 布林带标准差倍数，默认2.0
      - min_bars_between_signals: 连续信号之间最小间隔Bar数，默认3
      - log_level, collect_signals: 继承自BaseStrategy的日志与信号收集参数
    """
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('min_bars_between_signals', 3),
        ('log_level', BaseStrategy.LOG_LEVEL_INFO),
        ('collect_signals', True),
    )

    def __init__(self):
        super().__init__()
        # 初始化RSI指标
        self.rsi = bt.indicators.RSI(self.data.close,
                                     period=self.params.rsi_period)
        # 初始化布林带指标，使用默认中轨（均线）、上轨和下轨
        self.bb = bt.indicators.BollingerBands(self.data.close,
                                               period=self.params.bb_period,
                                               devfactor=self.params.bb_dev)
        # 用于控制连续信号间隔，防止频繁交易
        self.last_signal_bar = -self.params.min_bars_between_signals

    def next(self):
        super().next()  # 记录资产净值
        current_bar = len(self)
        # 如果距离上次信号还未超过设定的间隔，则不处理
        if current_bar - self.last_signal_bar < self.params.min_bars_between_signals:
            return

        # RSI信号判断
        rsi_buy = False
        rsi_sell = False
        if self.rsi[0] > self.params.rsi_oversold and self.rsi[
                -1] <= self.params.rsi_oversold:
            rsi_buy = True
        if self.rsi[0] < self.params.rsi_overbought and self.rsi[
                -1] >= self.params.rsi_overbought:
            rsi_sell = True

        # 布林带信号判断：
        # 买入：当收盘价从下方突破布林带下轨
        bb_buy = False
        if self.data.close[0] > self.bb.bot[0] and self.data.close[
                -1] <= self.bb.bot[-1]:
            bb_buy = True
        # 卖出：当收盘价从上方跌破布林带上轨
        bb_sell = False
        if self.data.close[0] < self.bb.top[0] and self.data.close[
                -1] >= self.bb.top[-1]:
            bb_sell = True

        # 如果当前无仓位且两个指标均给出买入信号，则买入
        if not self.position and rsi_buy and bb_buy:
            self.execute_buy()
        # 如果当前持仓且两个指标均给出卖出信号，则卖出
        elif self.position and rsi_sell and bb_sell:
            self.execute_sell()

    def execute_buy(self):
        price = self.data.close[0]
        size = self.calc_max_shares(price)
        if size <= 0:
            return
        self.order = self.buy(size=size)
        self.last_signal_bar = len(self)
        self.log(f'RSI+BB 买入信号: 价格={price:.2f}, 数量={size}')

    def execute_sell(self):
        price = self.data.close[0]
        size = self.position.size
        self.order = self.sell(size=size)
        self.last_signal_bar = len(self)
        self.log(f'RSI+BB 卖出信号: 价格={price:.2f}, 数量={size}')
