# """
# VWAP策略模块 - 基于成交量加权平均价格实现的交易策略
# """

# import backtrader as bt
# from .base_strategy import BaseStrategy
# from indicator.vwap import VWAP
# import numpy as np
# from datetime import time

# class VWAPStrategy(BaseStrategy):
#     """
#     基于成交量加权平均价(VWAP)的增强型交易策略

#     策略逻辑:
#     1. 计算VWAP及其标准差通道
#     2. 当价格从下方突破VWAP且成交量增加时买入
#     3. 价格跌破标准差下轨时止损
#     4. 价格触及标准差上轨时止盈
#     5. 在特定交易时段内交易，减少非交易时段的噪音
#     6. 增加趋势过滤，确保交易方向和整体趋势一致

#     改进点:
#     1. 使用标准差通道代替固定止盈止损比例
#     2. 增加交易时段过滤，只在高效市场时段交易
#     3. 添加趋势过滤，避免逆势交易
#     4. 优化成交量过滤，提高信号质量
#     5. 使用分批建仓和平仓，优化资金管理
#     """
#     params = (
#         ('vwap_period', 20),       # VWAP计算周期
#         ('volume_thresh', 1.5),    # 成交量阈值倍数
#         ('std_dev_mult', 2.0),     # 标准差通道倍数
#         ('reset_daily', True),     # 是否日内重置VWAP
#         ('use_std_channel', True), # 是否使用标准差通道
#         ('entry_pct', 0.7),        # 仓位入场比例
#         ('stop_loss', 0.05),       # 止损比例（非标准差通道模式）
#         ('take_profit', 0.10),     # 止盈比例（非标准差通道模式）
#         ('trailing_stop', 0.03),   # 跟踪止损比例
#         ('trend_period', 50),      # 趋势均线周期
#         ('use_time_filter', True), # 是否使用交易时段过滤
#         ('morning_start', time(9, 30)),  # 早盘开始时间
#         ('morning_end', time(11, 30)),   # 早盘结束时间
#         ('afternoon_start', time(13, 0)), # 午盘开始时间
#         ('afternoon_end', time(15, 0)),   # 午盘结束时间
#         # 继承BaseStrategy的参数
#         ('log_level', BaseStrategy.LOG_LEVEL_INFO),
#         ('collect_signals', True),
#     )

#     def __init__(self):
#         # 调用父类初始化
#         BaseStrategy.__init__(self)

#         # 创建自定义的VWAP指标
#         self.vwap = VWAP(
#             self.data,
#             period=self.params.vwap_period,
#             reset_daily=self.params.reset_daily,
#             std_dev_mult=self.params.std_dev_mult
#         )

#         # 创建成交量移动平均线指标
#         self.volume_ma = bt.indicators.SimpleMovingAverage(
#             self.data.volume, period=self.params.vwap_period)

#         # 交叉信号
#         self.price_cross_up_vwap = bt.indicators.CrossUp(self.data.close, self.vwap.vwap)
#         self.price_cross_down_vwap = bt.indicators.CrossDown(self.data.close, self.vwap.vwap)

#         # 趋势过滤器
#         self.trend_ma = bt.indicators.SimpleMovingAverage(
#             self.data.close, period=self.params.trend_period)

#         # 记录关键价格
#         self.entry_price = None
#         self.stop_price = None
#         self.target_price = None
#         self.peak_price = None

#         # 记录仓位状态
#         self.entry_complete = False
#         self.add_complete = False
#         self.max_position = 0

#     def next(self):
#         # 交易时段过滤
#         if self.params.use_time_filter and not self.is_trading_hours():
#             return

#         # 如果没有持仓
#         if not self.position:
#             # 当价格从下方突破VWAP且成交量放大时买入
#             if (self.price_cross_up_vwap[0] and
#                 self.data.volume[0] > self.volume_ma[0] * self.params.volume_thresh and
#                 self.data.close[0] > self.trend_ma[0]):  # 确保趋势向上

#                 price = self.data.close[0]

#                 # 计算仓位大小
#                 max_shares = self.calc_max_shares(price)
#                 entry_shares = int(max_shares * self.params.entry_pct)

#                 # 记录关键价格
#                 self.entry_price = price
#                 self.peak_price = price

#                 # 设置止盈止损价格
#                 if self.params.use_std_channel:
#                     # 使用标准差通道
#                     self.stop_price = self.vwap.vwap_lower[0]
#                     self.target_price = self.vwap.vwap_upper[0]
#                 else:
#                     # 使用固定比例
#                     self.stop_price = price * (1 - self.params.stop_loss)
#                     self.target_price = price * (1 + self.params.take_profit)

#                 if entry_shares > 0:
#                     self.log(f'买入信号(VWAP上穿): 价格={price:.2f}, 数量={entry_shares}, '
#                             f'VWAP={self.vwap.vwap[0]:.2f}, 成交量={self.data.volume[0]}')
#                     self.buy(size=entry_shares)
#                     self.max_position = max_shares
#                     self.entry_complete = True
#                     self.add_complete = False

#         # 如果有持仓
#         else:
#             current_position_size = self.position.size
#             current_price = self.data.close[0]

#             # 更新峰值价格（用于跟踪止损）
#             if current_price > self.peak_price:
#                 self.peak_price = current_price

#                 # 更新标准差通道止盈价格
#                 if self.params.use_std_channel:
#                     self.target_price = self.vwap.vwap_upper[0]
#                     self.stop_price = self.vwap.vwap_lower[0]

#             # 检查是否应该加仓（突破VWAP后价格进一步上涨10%）
#             if not self.add_complete and self.entry_complete and current_price > self.entry_price * 1.1:
#                 add_shares = self.max_position - current_position_size
#                 if add_shares > 0:
#                     self.log(f'加仓信号(价格上涨10%): 价格={current_price:.2f}, 加仓数量={add_shares}')
#                     self.buy(size=add_shares)
#                     self.add_complete = True
#                     return

#             # 卖出策略：标准差通道或VWAP下穿

#             # 当价格从上方跌破VWAP时卖出
#             if self.price_cross_down_vwap[0]:
#                 self.log(f'卖出信号(VWAP下穿): 价格={current_price:.2f}, '
#                         f'VWAP={self.vwap.vwap[0]:.2f}, 持仓={current_position_size}')
#                 self.close()
#                 return

#             # 止损
#             if current_price < self.stop_price:
#                 self.log(f'卖出信号(止损): 价格={current_price:.2f}, 止损价={self.stop_price:.2f}, 持仓={current_position_size}')
#                 self.close()
#                 return

#             # 止盈
#             if current_price > self.target_price:
#                 self.log(f'卖出信号(止盈): 价格={current_price:.2f}, 止盈价={self.target_price:.2f}, 持仓={current_position_size}')
#                 self.close()
#                 return

#             # 跟踪止损
#             if current_price < self.peak_price * (1 - self.params.trailing_stop) and current_price > self.entry_price:
#                 self.log(f'卖出信号(跟踪止损): 价格={current_price:.2f}, 峰值={self.peak_price:.2f}, 持仓={current_position_size}')
#                 self.close()
#                 return

#     def is_trading_hours(self):
#         """判断当前是否在交易时段内"""
#         current_time = self.data.datetime.time(0)

#         # 检查是否在早盘
#         morning_session = (current_time >= self.params.morning_start and
#                           current_time <= self.params.morning_end)

#         # 检查是否在午盘
#         afternoon_session = (current_time >= self.params.afternoon_start and
#                             current_time <= self.params.afternoon_end)

#         return morning_session or afternoon_session

#     def notify_order(self, order):
#         """处理订单状态变化"""
#         super(VWAPStrategy, self).notify_order(order)

#         # 如果订单完成，更新入场价格
#         if order.status == order.Completed and order.isbuy():
#             self.entry_price = order.executed.price
#             self.peak_price = max(self.peak_price, self.entry_price)

#             # 更新止损价格
#             if not self.params.use_std_channel:
#                 self.stop_price = self.entry_price * (1 - self.params.stop_loss)
#                 self.target_price = self.entry_price * (1 + self.params.take_profit)

#     def get_strategy_name(self):
#         """返回策略名称"""
#         return "增强型VWAP策略"

#     def get_strategy_description(self):
#         """返回策略描述"""
#         return f"""增强型VWAP策略

#         参数：
#         - VWAP周期: {self.params.vwap_period}天
#         - 标准差倍数: {self.params.std_dev_mult}
#         - 使用标准差通道: {self.params.use_std_channel}
#         - 日内重置VWAP: {self.params.reset_daily}
#         - 交易时段过滤: {self.params.use_time_filter}

#         策略逻辑:
#         1. 计算VWAP及其标准差通道
#         2. 当价格从下方突破VWAP且成交量增加时买入
#         3. 使用VWAP标准差通道进行止盈止损
#         4. 添加趋势过滤，确保交易方向和整体趋势一致
#         5. 在特定交易时段内交易，减少非交易时段的噪音
#         """
"""
VWAP策略模块 - 基于成交量加权平均价格实现的交易策略
"""
from backtrade.strategy.base_strategy import BaseStrategy
from backtrade.indicator.vwap import VWAP

import backtrader as bt
import numpy as np
from datetime import time


class VWAPStrategy(BaseStrategy):
    """
    基于成交量加权平均价(VWAP)的增强型交易策略
    
    策略逻辑:
    1. 计算VWAP及其标准差通道
    2. 当价格从下方突破VWAP且成交量增加时买入
    3. 价格跌破标准差下轨时止损
    4. 价格触及标准差上轨时止盈
    5. 在特定交易时段内交易，减少非交易时段的噪音
    6. 增加趋势过滤，确保交易方向和整体趋势一致
    """
    params = (
        ('vwap_period', 20),
        ('volume_thresh', 1.5),
        ('std_dev_mult', 2.0),
        ('reset_daily', True),
        ('use_std_channel', True),
        ('entry_pct', 0.7),
        ('stop_loss', 0.05),
        ('take_profit', 0.10),
        ('trailing_stop', 0.03),
        ('trend_period', 50),
        ('use_time_filter', True),
        ('morning_start', time(9, 30)),
        ('morning_end', time(11, 30)),
        ('afternoon_start', time(13, 0)),
        ('afternoon_end', time(15, 0)),
        ('log_level', BaseStrategy.LOG_LEVEL_INFO),
        ('collect_signals', True),
    )

    def __init__(self):
        super().__init__()

        # 自定义VWAP指标
        self.vwap = VWAP(self.data,
                         period=self.params.vwap_period,
                         reset_daily=self.params.reset_daily,
                         std_dev_mult=self.params.std_dev_mult)

        # 成交量均线
        self.volume_ma = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=self.params.vwap_period)

        # 价格穿越检测
        self.price_cross_up_vwap = bt.indicators.CrossUp(
            self.data.close, self.vwap.vwap)
        self.price_cross_down_vwap = bt.indicators.CrossDown(
            self.data.close, self.vwap.vwap)

        # 趋势过滤
        self.trend_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.trend_period)

        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.peak_price = None

        self.entry_complete = False
        self.add_complete = False
        self.max_position = 0

    def next(self):
        # 交易时段过滤
        if self.params.use_time_filter and not self.is_trading_hours():
            return

        if not self.position:
            # 当价格从下方突破VWAP + 成交量放大 + 趋势向上
            if (self.price_cross_up_vwap[0]
                    and (self.data.volume[0]
                         > self.volume_ma[0] * self.params.volume_thresh)
                    and (self.data.close[0] > self.trend_ma[0])):

                price = self.data.close[0]
                max_shares = self.calc_max_shares(price)
                entry_shares = int(max_shares * self.params.entry_pct)

                if entry_shares > 0:
                    self.entry_price = price
                    self.peak_price = price

                    if self.params.use_std_channel:
                        self.stop_price = self.vwap.vwap_lower[0]
                        self.target_price = self.vwap.vwap_upper[0]
                    else:
                        self.stop_price = price * (1 - self.params.stop_loss)
                        self.target_price = price * (1 +
                                                     self.params.take_profit)

                    self.log(
                        f'买入信号(VWAP上穿): 价格={price:.2f}, 数量={entry_shares}, '
                        f'VWAP={self.vwap.vwap[0]:.2f}, 成交量={self.data.volume[0]}'
                    )

                    self.buy(size=entry_shares)
                    self.max_position = max_shares
                    self.entry_complete = True
                    self.add_complete = False

        else:
            current_price = self.data.close[0]

            # 更新峰值
            if current_price > self.peak_price:
                self.peak_price = current_price
                if self.params.use_std_channel:
                    self.target_price = self.vwap.vwap_upper[0]
                    self.stop_price = self.vwap.vwap_lower[0]

            # 加仓（突破后再上涨10%）
            if (not self.add_complete and self.entry_complete
                    and current_price > self.entry_price * 1.1):

                add_shares = self.max_position - self.position.size
                if add_shares > 0:
                    self.log(
                        f'加仓信号(价格上涨10%): 价格={current_price:.2f}, 加仓数量={add_shares}'
                    )
                    self.buy(size=add_shares)
                    self.add_complete = True
                    return

            # 卖出条件
            # 1. 价格从上方跌破VWAP
            if self.price_cross_down_vwap[0]:
                self.log(
                    f'卖出信号(VWAP下穿): 价格={current_price:.2f}, '
                    f'VWAP={self.vwap.vwap[0]:.2f}, 持仓={self.position.size}')
                self.close()
                return

            # 2. 止损
            if current_price < self.stop_price:
                self.log(
                    f'卖出信号(止损): 价格={current_price:.2f}, 止损价={self.stop_price:.2f}, 持仓={self.position.size}'
                )
                self.close()
                return

            # 3. 止盈
            if current_price > self.target_price:
                self.log(
                    f'卖出信号(止盈): 价格={current_price:.2f}, 止盈价={self.target_price:.2f}, 持仓={self.position.size}'
                )
                self.close()
                return

            # 4. 跟踪止损
            if current_price < self.peak_price * (
                    1 - self.params.trailing_stop
            ) and current_price > self.entry_price:
                self.log(
                    f'卖出信号(跟踪止损): 价格={current_price:.2f}, 峰值={self.peak_price:.2f}, 持仓={self.position.size}'
                )
                self.close()
                return

    def is_trading_hours(self):
        current_time = self.data.datetime.time(0)

        morning_session = (current_time >= self.params.morning_start
                           and current_time <= self.params.morning_end)

        afternoon_session = (current_time >= self.params.afternoon_start
                             and current_time <= self.params.afternoon_end)

        return morning_session or afternoon_session

    def notify_order(self, order):
        super().notify_order(order)

        if order.status == order.Completed and order.isbuy():
            self.entry_price = order.executed.price
            self.peak_price = max(self.peak_price, self.entry_price)

            if not self.params.use_std_channel:
                self.stop_price = self.entry_price * (1 -
                                                      self.params.stop_loss)
                self.target_price = self.entry_price * (
                    1 + self.params.take_profit)

    def get_strategy_name(self):
        return "增强型VWAP策略"

    def get_strategy_description(self):
        return f"""增强型VWAP策略
        
        参数：
        - VWAP周期: {self.params.vwap_period}天
        - 标准差倍数: {self.params.std_dev_mult}
        - 使用标准差通道: {self.params.use_std_channel}
        - 日内重置VWAP: {self.params.reset_daily}
        - 交易时段过滤: {self.params.use_time_filter}
        """
