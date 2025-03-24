from backtrade.indicator.obv import OnBalanceVolume
from backtrade.strategy.base_strategy import BaseStrategy

import backtrader as bt
import numpy as np


class OBVStrategy(BaseStrategy):
    """
    增强型OBV(On-Balance Volume)交易策略
    
    策略逻辑:
    1. 买入条件：
       - OBV上穿其EMA且价格在其EMA上方
       - 或者出现正背离（价格下跌但OBV上升）
       - 信号需经过趋势确认和成交量确认
    
    2. 卖出条件：
       - OBV下穿其EMA且价格在其EMA下方
       - 或者出现负背离（价格上涨但OBV下降）
       - 或者触发止盈止损
       
    改进点：
    1. 增加趋势过滤机制
    2. 增加成交量确认要求
    3. 动态止盈止损
    4. 分批建仓和平仓
    5. 增加信号确认和过滤
    """

    params = (
        ('obv_ema_period', 20),  # OBV均线周期
        ('price_ema_period', 20),  # 价格均线周期
        ('trend_period', 50),  # 趋势判断周期
        ('volume_ratio_min', 1.2),  # 最小成交量比例
        ('stop_loss', 0.05),  # 止损比例
        ('take_profit', 0.15),  # 止盈比例
        ('trailing_stop', 0.03),  # 跟踪止损比例
        ('use_position_sizing', True),  # 是否使用仓位管理
        ('risk_per_trade', 0.02),  # 每笔交易风险（资金百分比）
        ('entry_fraction', 0.5),  # 初始建仓比例
        ('add_pos_threshold', 0.05),  # 加仓阈值（盈利比例）
        ('log_level', BaseStrategy.LOG_LEVEL_INFO),
        ('collect_signals', True),
    )

    def __init__(self):
        super(OBVStrategy, self).__init__()

        # 计算OBV指标
        self.obv = OnBalanceVolume(
            self.data,
            use_volume_at_first_bar=False,
            normalize=True,
            smooth_period=5,
        )

        # OBV的 EMA
        self.obv_ema = bt.indicators.ExponentialMovingAverage(
            self.obv.obv_ema, period=self.params.obv_ema_period)

        # 价格的 EMA
        self.price_ema = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.params.price_ema_period)

        # 长期趋势
        self.trend_ema = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.params.trend_period)

        # 成交量指标
        self.volume_ratio = (
            bt.indicators.SimpleMovingAverage(self.data.volume, period=5) /
            bt.indicators.SimpleMovingAverage(self.data.volume, period=20))

        # 止盈止损相关
        self.stop_price = None
        self.target_price = None
        self.peak_price = None

        self.order = None
        self.entry_price = None
        self.position_size = 0
        self.max_position_size = 0
        self.entry_executed = False
        self.add_position_executed = False

    def next(self):
        super().next()  # 记录资产净值
        # 若有挂单则不处理
        if self.order:
            return

        today_price = self.data.close[0]
        yesterday_price = self.data.close[-1]
        today_obv = self.obv.obv[0]
        yesterday_obv = self.obv.obv[-1]

        # 价格/OBV 方向
        price_up = (today_price > yesterday_price)
        obv_up = (today_obv > yesterday_obv)

        # 背离
        positive_divergence = (not price_up) and obv_up and (
            today_price < self.price_ema[0])
        negative_divergence = price_up and (not obv_up) and (
            today_price > self.price_ema[0])

        # OBV 与 OBV EMA 的穿越
        obv_crossover = (today_obv > self.obv_ema[0]
                         and yesterday_obv <= self.obv_ema[-1])
        obv_crossunder = (today_obv < self.obv_ema[0]
                          and yesterday_obv >= self.obv_ema[-1])

        # 价格与价格EMA
        price_above_ema = (today_price > self.price_ema[0])
        price_below_ema = (today_price < self.price_ema[0])

        # 长期趋势
        uptrend = (today_price > self.trend_ema[0])
        downtrend = (today_price < self.trend_ema[0])

        # 成交量有效性
        volume_valid = (self.volume_ratio[0] >= self.params.volume_ratio_min)

        # 多头开仓
        if not self.position:
            valid_buy_signal = (((obv_crossover and price_above_ema)
                                 or positive_divergence) and uptrend
                                and volume_valid)
            if valid_buy_signal:
                self.log(
                    f'买入信号: OBV穿越={obv_crossover}, 价格位置={price_above_ema}, '
                    f'正背离={positive_divergence}, 趋势={uptrend}, 成交量={volume_valid}'
                )

                if self.params.use_position_sizing:
                    risk_amount = self.broker.get_value(
                    ) * self.params.risk_per_trade
                    stop_price = today_price * (1 - self.params.stop_loss)
                    per_share_risk = today_price - stop_price
                    position_size = risk_amount / per_share_risk if per_share_risk > 0 else 0
                    size = int(position_size * self.params.entry_fraction)
                else:
                    size = self.calc_max_shares(today_price)

                size = max(1, size)

                if size > 0:
                    self.entry_price = today_price
                    self.position_size = size
                    self.max_position_size = int(
                        position_size
                    ) if self.params.use_position_sizing else size

                    self.stop_price = self.entry_price * (
                        1 - self.params.stop_loss)
                    self.target_price = self.entry_price * (
                        1 + self.params.take_profit)
                    self.peak_price = self.entry_price

                    self.log(
                        f'执行买入: 价格={today_price:.2f}, 数量={size}, '
                        f'止损={self.stop_price:.2f}, 止盈={self.target_price:.2f}'
                    )

                    self.order = self.buy(size=size)
                    self.entry_executed = True
                    self.add_position_executed = False

        else:
            # 更新峰值
            if today_price > self.peak_price:
                self.peak_price = today_price

            # 加仓
            if (self.params.use_position_sizing
                    and today_price >= self.entry_price *
                (1 + self.params.add_pos_threshold)
                    and not self.add_position_executed
                    and self.position.size < self.max_position_size):

                add_size = self.max_position_size - self.position.size
                if add_size > 0:
                    self.log(
                        f'执行加仓: 价格={today_price:.2f}, 加仓数量={add_size}, '
                        f'原仓位={self.position.size}, 新仓位={self.position.size + add_size}'
                    )
                    self.order = self.buy(size=add_size)
                    self.add_position_executed = True

            # 止损
            elif today_price <= self.stop_price:
                self.log(
                    f'触发止损: 价格={today_price:.2f}, 止损价={self.stop_price:.2f}, 仓位={self.position.size}'
                )
                self.order = self.sell(size=self.position.size)

            # 跟踪止损
            elif today_price <= self.peak_price * (
                    1 - self.params.trailing_stop
            ) and today_price > self.entry_price:
                self.log(
                    f'触发跟踪止损: 当前价格={today_price:.2f}, 峰值={self.peak_price:.2f}, '
                    f'回撤比例={1 - today_price/self.peak_price:.2%}, 仓位={self.position.size}'
                )
                self.order = self.sell(size=self.position.size)

            # 止盈
            elif today_price >= self.target_price:
                self.log(
                    f'触发止盈: 价格={today_price:.2f}, 止盈价={self.target_price:.2f}, 仓位={self.position.size}'
                )
                self.order = self.sell(size=self.position.size)

            # OBV信号下穿 + 趋势向下
            elif ((obv_crossunder and price_below_ema)
                  or negative_divergence) and downtrend:
                self.log(
                    f'卖出信号: OBV穿越={obv_crossunder}, 价格位置={price_below_ema}, '
                    f'负背离={negative_divergence}, 趋势={downtrend}, 仓位={self.position.size}'
                )
                self.order = self.sell(size=self.position.size)

    def notify_order(self, order):
        super(OBVStrategy, self).notify_order(order)

        if order.status in [
                order.Completed, order.Canceled, order.Margin, order.Rejected
        ]:
            self.order = None

            # 如果是买单完成，更新 entry_price、止损止盈等
            if order.isbuy() and order.status == order.Completed:
                self.entry_price = order.executed.price
                self.stop_price = self.entry_price * (1 -
                                                      self.params.stop_loss)
                self.target_price = self.entry_price * (
                    1 + self.params.take_profit)
                self.peak_price = self.entry_price

    def stop(self):
        super(OBVStrategy, self).stop()
        self.log(
            f'OBV策略参数: OBV EMA周期={self.params.obv_ema_period}, 价格EMA周期={self.params.price_ema_period}, '
            f'止损={self.params.stop_loss:.2%}, 止盈={self.params.take_profit:.2%}'
        )

    def get_strategy_name(self):
        return "增强型OBV策略"

    def get_strategy_description(self):
        return f"""增强型OBV(On-Balance Volume)交易策略
        
        参数:
        - OBV EMA周期: {self.params.obv_ema_period}
        - 价格EMA周期: {self.params.price_ema_period}
        - 趋势周期: {self.params.trend_period}
        - 止损: {self.params.stop_loss:.2%}
        - 止盈: {self.params.take_profit:.2%}
        - 跟踪止损: {self.params.trailing_stop:.2%}
        - 使用仓位管理: {self.params.use_position_sizing}
        - 每笔交易风险: {self.params.risk_per_trade:.2%}
        """
