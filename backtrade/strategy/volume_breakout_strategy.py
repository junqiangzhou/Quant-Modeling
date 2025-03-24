from .base_strategy import BaseStrategy
import backtrader as bt
import numpy as np


class VolumeBreakoutStrategy(BaseStrategy):
    """
    增强型交易量突破策略
    
    基本思想：监控成交量的突然放大，结合价格走势，捕捉潜在的突破机会
    
    改进点：
    1. 增加形态确认
    2. 动态成交量阈值
    3. 趋势一致性过滤
    4. 波动率过滤
    5. 优化止盈止损（跟踪止损、自适应止盈）
    """
    params = (
        ('volume_period', 20),
        ('volume_mult', 2.0),
        ('exit_bars', 5),
        ('stop_loss', 0.05),
        ('take_profit', 0.10),
        ('trailing_stop', 0.03),
        ('trend_period', 50),
        ('use_atr_stops', True),
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('require_price_confirm', True),
        ('price_confirm_pct', 1.0),
        ('volatility_filter', True),
        ('min_volatility', 0.5),
        ('max_volatility', 3.0),
        ('dynamic_volume', True),
        ('adaptive_exit', True),
        ('log_level', BaseStrategy.LOG_LEVEL_INFO),
        ('collect_signals', True),
    )

    def __init__(self):
        super().__init__()

        self.volume_ma = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=self.params.volume_period)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.trend_ma = bt.indicators.EMA(self.data.close,
                                          period=self.params.trend_period)
        self.highest = bt.indicators.Highest(self.data.high,
                                             period=self.params.volume_period)

        if self.params.dynamic_volume:
            self.volume_std = bt.indicators.StdDev(
                self.data.volume, period=self.params.volume_period)
            self.dynamic_volume_thresh = lambda: self.volume_ma[
                0] + self.volume_std[0] * 2
        else:
            self.dynamic_volume_thresh = lambda: self.volume_ma[
                0] * self.params.volume_mult

        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.peak_price = None
        self.entry_bar = None

    def next(self):
        if not self.position:
            if self.is_valid_entry():
                price = self.data.close[0]
                max_shares = self.calc_max_shares(price)

                if max_shares > 0:
                    self.entry_price = price
                    self.peak_price = price
                    self.entry_bar = len(self)

                    if self.params.use_atr_stops:
                        self.stop_price = price - self.atr[
                            0] * self.params.atr_multiplier
                        self.target_price = price + self.atr[
                            0] * self.params.atr_multiplier * 2
                    else:
                        self.stop_price = price * (1 - self.params.stop_loss)
                        self.target_price = price * (1 +
                                                     self.params.take_profit)

                    self.log(
                        f'买入信号: 价格={price:.2f}, 数量={max_shares}, '
                        f'交易量={self.data.volume[0]:.0f}, 平均交易量={self.volume_ma[0]:.0f}, '
                        f'止损={self.stop_price:.2f}, 止盈={self.target_price:.2f}'
                    )

                    self.buy(size=max_shares)
                else:
                    self.log(
                        f'资金不足无法买入: 价格={price:.2f}, 可用资金={self.broker.getcash():.2f}'
                    )

        else:
            current_price = self.data.close[0]

            # 更新峰值价格
            if current_price > self.peak_price:
                self.peak_price = current_price
                if self.params.adaptive_exit:
                    price_gain = (current_price / self.entry_price - 1)
                    if price_gain > 0.05:
                        self.target_price = max(self.target_price,
                                                current_price * 0.98)

            exit_signal = self.check_exit_signals(current_price)
            if exit_signal:
                self.log(
                    f'卖出信号({exit_signal}): 价格={current_price:.2f}, 持仓数量={self.position.size}'
                )
                self.close()

    def is_valid_entry(self):
        """检查是否满足入场条件"""
        volume_breakout = (self.data.volume[0] > self.dynamic_volume_thresh())
        if not volume_breakout:
            return False

        trend_up = (self.data.close[0] > self.trend_ma[0])

        price_confirmed = True
        if self.params.require_price_confirm:
            near_high = (self.data.close[0] >= self.highest[-1] *
                         (1 - self.params.price_confirm_pct / 100))
            price_up = (self.data.close[0] > self.data.close[-1])
            price_confirmed = (near_high and price_up)

        volatility_ok = True
        if self.params.volatility_filter:
            current_volatility = (self.atr[0] / self.data.close[0]) * 100
            volatility_ok = (self.params.min_volatility <= current_volatility
                             <= self.params.max_volatility)

        valid_entry = volume_breakout and trend_up and price_confirmed and volatility_ok

        if volume_breakout and not valid_entry:
            self.log(
                f'成交量突破但不满足其他条件: 趋势={trend_up}, '
                f'价格确认={price_confirmed}, 波动率适中={volatility_ok}',
                level=self.LOG_LEVEL_DEBUG)

        return valid_entry

    def check_exit_signals(self, current_price):
        """检查是否满足出场条件"""
        if self.params.exit_bars > 0 and len(self) >= (self.entry_bar +
                                                       self.params.exit_bars):
            return '时间退出'

        if current_price < self.stop_price:
            return '止损'

        if current_price > self.target_price:
            return '止盈'

        if (self.peak_price > self.entry_price
                and current_price < self.peak_price *
            (1 - self.params.trailing_stop)):
            return '跟踪止损'

        return None

    def notify_order(self, order):
        super().notify_order(order)

        if order.status == order.Completed and order.isbuy():
            self.entry_price = order.executed.price
            self.peak_price = max(self.entry_price, self.peak_price or 0)

            if self.params.use_atr_stops:
                self.stop_price = self.entry_price - self.atr[
                    0] * self.params.atr_multiplier
                self.target_price = self.entry_price + self.atr[
                    0] * self.params.atr_multiplier * 2
            else:
                self.stop_price = self.entry_price * (1 -
                                                      self.params.stop_loss)
                self.target_price = self.entry_price * (
                    1 + self.params.take_profit)

    def get_strategy_name(self):
        return "增强型成交量突破策略"

    def get_strategy_description(self):
        return f"""增强型成交量突破策略
        
        参数：
        - 成交量周期: {self.params.volume_period}天
        - {'动态' if self.params.dynamic_volume else '静态'}成交量阈值
        - 趋势周期: {self.params.trend_period}天
        - 持有周期: {self.params.exit_bars if self.params.exit_bars > 0 else '不限'}天
        - 使用ATR止损: {self.params.use_atr_stops} (倍数: {self.params.atr_multiplier})
        - 跟踪止损: {self.params.trailing_stop*100}%
        - 自适应止盈: {self.params.adaptive_exit}
        """
