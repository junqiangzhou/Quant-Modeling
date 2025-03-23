# import backtrader as bt
# import numpy as np
# from datetime import time

# class VWAP(bt.Indicator):
#     """
#     改进的VWAP (Volume Weighted Average Price) 指标实现

#     计算公式: VWAP = Σ(成交价 × 成交量) / Σ(成交量)

#     改进点:
#     1. 支持标准差通道计算 - 用于识别超买超卖区域
#     2. 支持日内VWAP重置 - 符合日内交易惯例
#     3. 增加滑动窗口VWAP - 限制计算范围，更敏感
#     4. 优化内存使用和性能 - 使用队列结构而非列表
#     """
#     lines = ('vwap', 'vwap_upper', 'vwap_lower',)
#     params = (
#         ('period', 20),         # 滑动窗口周期
#         ('reset_daily', False), # 是否每日重置VWAP
#         ('use_typical', True),  # 是否使用典型价格
#         ('std_dev_mult', 2.0),  # 标准差倍数
#     )

#     def __init__(self):
#         # 初始化累计值
#         self.cum_vol = 0
#         self.cum_vol_price = 0

#         # 存储历史数据
#         self.price_volume_queue = []  # 存储 (price, volume) 对
#         self.daily_prices = []        # 存储每日价格用于标准差计算

#         # 添加最小周期需求
#         self.addminperiod(1)

#         # 记录上一个日期，用于检测日期变化
#         self.last_date = None

#     def next(self):
#         # 获取当前日期和时间
#         current_datetime = self.data.datetime.datetime(0)
#         current_date = current_datetime.date()

#         # 如果启用日内重置且日期发生变化
#         if self.p.reset_daily and (self.last_date is None or current_date != self.last_date):
#             self.reset_vwap()
#             self.last_date = current_date

#         # 计算当前价格
#         if self.p.use_typical:
#             # 典型价格 (TP) = (High + Low + Close) / 3
#             current_price = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3
#         else:
#             # 使用收盘价
#             current_price = self.data.close[0]

#         # 获取成交量
#         current_vol = self.data.volume[0]

#         # 计算价格*成交量
#         current_vol_price = current_vol * current_price

#         # 添加到队列
#         self.price_volume_queue.append((current_price, current_vol))

#         # 如果超过周期长度，移除最早的数据
#         if len(self.price_volume_queue) > self.params.period:
#             old_price, old_vol = self.price_volume_queue.pop(0)
#             self.cum_vol -= old_vol
#             self.cum_vol_price -= old_price * old_vol

#         # 更新累计值
#         self.cum_vol += current_vol
#         self.cum_vol_price += current_vol_price

#         # 计算VWAP
#         if self.cum_vol > 0:
#             self.lines.vwap[0] = self.cum_vol_price / self.cum_vol
#         else:
#             # 无成交量时使用当前价格
#             self.lines.vwap[0] = current_price

#         # 存储价格用于标准差计算
#         self.daily_prices.append(current_price)
#         if len(self.daily_prices) > self.params.period:
#             self.daily_prices.pop(0)

#         # 计算标准差通道
#         if len(self.daily_prices) > 1:
#             std_dev = np.std(self.daily_prices)
#             self.lines.vwap_upper[0] = self.lines.vwap[0] + self.params.std_dev_mult * std_dev
#             self.lines.vwap_lower[0] = self.lines.vwap[0] - self.params.std_dev_mult * std_dev
#         else:
#             # 数据不足时使用VWAP值
#             self.lines.vwap_upper[0] = self.lines.vwap[0]
#             self.lines.vwap_lower[0] = self.lines.vwap[0]

#     def reset_vwap(self):
#         """重置VWAP计算，用于日内重置"""
#         self.cum_vol = 0
#         self.cum_vol_price = 0
#         self.price_volume_queue = []
#         self.daily_prices = []

import backtrader as bt
import numpy as np
from datetime import time


class VWAP(bt.Indicator):
    """
    改进的VWAP (Volume Weighted Average Price) 指标实现
    
    计算公式: VWAP = Σ(成交价 × 成交量) / Σ(成交量)
    
    改进点:
    1. 支持标准差通道计算 - 用于识别超买超卖区域
    2. 支持日内VWAP重置 - 符合日内交易惯例
    3. 增加滑动窗口VWAP - 限制计算范围，更敏感
    4. 优化内存使用和性能 - 使用队列结构而非列表
    """
    lines = (
        'vwap',
        'vwap_upper',
        'vwap_lower',
    )
    params = (
        ('period', 20),  # 滑动窗口周期
        ('reset_daily', False),  # 是否每日重置VWAP
        ('use_typical', True),  # 是否使用典型价格
        ('std_dev_mult', 2.0),  # 标准差倍数
    )

    def __init__(self):
        self.cum_vol = 0
        self.cum_vol_price = 0

        # (price, volume) 队列
        self.price_volume_queue = []
        # 用于标准差计算
        self.daily_prices = []

        # 记录上一个交易日
        self.last_date = None

        self.addminperiod(1)

    def next(self):
        current_datetime = self.data.datetime.datetime(0)
        current_date = current_datetime.date()

        # 日内重置
        if self.p.reset_daily and (self.last_date is None
                                   or current_date != self.last_date):
            self.reset_vwap()
            self.last_date = current_date

        # 计算当前 bar 的价格
        if self.p.use_typical:
            current_price = (self.data.high[0] + self.data.low[0] +
                             self.data.close[0]) / 3
        else:
            current_price = self.data.close[0]

        current_vol = self.data.volume[0]
        current_vol_price = current_vol * current_price

        # 添加到队列
        self.price_volume_queue.append((current_price, current_vol))

        # 超过周期长度则弹出最早的
        if len(self.price_volume_queue) > self.params.period:
            old_price, old_vol = self.price_volume_queue.pop(0)
            self.cum_vol -= old_vol
            self.cum_vol_price -= old_price * old_vol

        # 更新累计值
        self.cum_vol += current_vol
        self.cum_vol_price += current_vol_price

        # 计算VWAP
        if self.cum_vol > 0:
            self.lines.vwap[0] = self.cum_vol_price / self.cum_vol
        else:
            self.lines.vwap[0] = current_price

        # 维护 daily_prices 用于计算 std
        self.daily_prices.append(current_price)
        if len(self.daily_prices) > self.params.period:
            self.daily_prices.pop(0)

        # 计算上/下轨
        if len(self.daily_prices) > 1:
            std_dev = np.std(self.daily_prices)
            self.lines.vwap_upper[
                0] = self.lines.vwap[0] + self.params.std_dev_mult * std_dev
            self.lines.vwap_lower[
                0] = self.lines.vwap[0] - self.params.std_dev_mult * std_dev
        else:
            self.lines.vwap_upper[0] = self.lines.vwap[0]
            self.lines.vwap_lower[0] = self.lines.vwap[0]

    def reset_vwap(self):
        """重置VWAP，用于日内场景"""
        self.cum_vol = 0
        self.cum_vol_price = 0
        self.price_volume_queue = []
        self.daily_prices = []
