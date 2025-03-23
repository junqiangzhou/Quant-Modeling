# import backtrader as bt
# import numpy as np

# class MFI(bt.Indicator):
#     """
#     改进版 Money Flow Index (MFI) 指标

#     计算流程：
#       1. 计算典型价格：TP = (High + Low + Close) / 3
#       2. 计算资金流：MF = TP * Volume
#       3. 判断正负资金流：若TP大于前一期TP，则为正资金流，否则为负资金流
#       4. 计算过去周期内正、负资金流的累加和，并计算资金流比率 MFR
#       5. 计算 MFI = 100 - 100/(1 + MFR)

#     改进点：
#       1. 添加背离检测功能 - 识别价格与MFI的不一致走势
#       2. 添加超买超卖区域定义 - 可自定义阈值
#       3. 添加趋势状态指示 - 判断当前MFI的趋势状态
#       4. 添加信号生成功能 - 基于MFI生成买入/卖出信号
#     """
#     lines = ('mfi', 'signal',)
#     params = (
#         ('period', 14),          # MFI周期
#         ('oversold', 20),        # 超卖阈值
#         ('overbought', 80),      # 超买阈值
#         ('signal_period', 9),    # 信号线周期
#         ('divergence_period', 10), # 背离检测周期
#     )

#     def __init__(self):
#         # 计算典型价格
#         tp = (self.data.high + self.data.low + self.data.close) / 3.0
#         mf = tp * self.data.volume
#         tp_prev = tp(-1)

#         # 判断正负资金流
#         pos_flow = bt.If(tp > tp_prev, mf, 0.0)
#         neg_flow = bt.If(tp < tp_prev, mf, 0.0)

#         # 累计正负资金流，避免除零
#         pos_flow_sum = bt.indicators.SumN(pos_flow, period=self.p.period)
#         neg_flow_sum = bt.indicators.SumN(neg_flow, period=self.p.period)

#         # 计算资金流比率和MFI
#         mfr = pos_flow_sum / (neg_flow_sum + 1e-10)
#         self.lines.mfi = 100 - (100 / (1 + mfr))

#         # 计算信号线 (MFI的移动平均)
#         self.lines.signal = bt.indicators.EMA(self.lines.mfi, period=self.p.signal_period)

#         # 记录最近的极值用于背离检测
#         self.price_highs = []  # 价格高点列表
#         self.price_lows = []   # 价格低点列表
#         self.mfi_highs = []    # MFI高点列表
#         self.mfi_lows = []     # MFI低点列表

#         # 添加最小周期
#         self.addminperiod(max(self.p.period, self.p.divergence_period) + 1)

#     def next(self):
#         # 更新极值点 (在每个bar执行)
#         self.update_swing_points()

#     def update_swing_points(self):
#         """更新MFI和价格的极值点"""
#         # 检查是否形成价格高点
#         if self.is_high_point(self.data.close, self.p.divergence_period):
#             self.price_highs.append((len(self), self.data.close[0]))
#             # 同时记录对应的MFI值
#             self.mfi_highs.append((len(self), self.lines.mfi[0]))
#             # 保留最近的3个高点
#             if len(self.price_highs) > 3:
#                 self.price_highs.pop(0)
#                 self.mfi_highs.pop(0)

#         # 检查是否形成价格低点
#         if self.is_low_point(self.data.close, self.p.divergence_period):
#             self.price_lows.append((len(self), self.data.close[0]))
#             # 同时记录对应的MFI值
#             self.mfi_lows.append((len(self), self.lines.mfi[0]))
#             # 保留最近的3个低点
#             if len(self.price_lows) > 3:
#                 self.price_lows.pop(0)
#                 self.mfi_lows.pop(0)

#     def is_high_point(self, series, lookback=5):
#         """检查当前值是否为局部高点"""
#         middle_idx = lookback // 2
#         if len(series) < lookback:
#             return False

#         # 提取lookback周期内的数据
#         window = [series[-i] for i in range(lookback)]

#         # 检查中间点是否为最大值
#         return window[middle_idx] == max(window)

#     def is_low_point(self, series, lookback=5):
#         """检查当前值是否为局部低点"""
#         middle_idx = lookback // 2
#         if len(series) < lookback:
#             return False

#         # 提取lookback周期内的数据
#         window = [series[-i] for i in range(lookback)]

#         # 检查中间点是否为最小值
#         return window[middle_idx] == min(window)

#     def check_bullish_divergence(self):
#         """检查看涨背离：价格创新低，但MFI未创新低"""
#         if len(self.price_lows) < 2 or len(self.mfi_lows) < 2:
#             return False

#         # 获取最近两个低点
#         (_, curr_price_low), (_, prev_price_low) = self.price_lows[-2:]
#         (_, curr_mfi_low), (_, prev_mfi_low) = self.mfi_lows[-2:]

#         # 判断是否为看涨背离
#         return curr_price_low < prev_price_low and curr_mfi_low > prev_mfi_low

#     def check_bearish_divergence(self):
#         """检查看跌背离：价格创新高，但MFI未创新高"""
#         if len(self.price_highs) < 2 or len(self.mfi_highs) < 2:
#             return False

#         # 获取最近两个高点
#         (_, curr_price_high), (_, prev_price_high) = self.price_highs[-2:]
#         (_, curr_mfi_high), (_, prev_mfi_high) = self.mfi_highs[-2:]

#         # 判断是否为看跌背离
#         return curr_price_high > prev_price_high and curr_mfi_high < prev_mfi_high

#     def get_zone(self):
#         """获取当前MFI的区域状态"""
#         if self.lines.mfi[0] > self.p.overbought:
#             return "超买"
#         elif self.lines.mfi[0] < self.p.oversold:
#             return "超卖"
#         else:
#             return "中性"

#     def get_trend(self):
#         """获取当前MFI的趋势状态"""
#         # 简单地使用MFI与其信号线的关系判断趋势
#         if self.lines.mfi[0] > self.lines.signal[0]:
#             return "上升"
#         elif self.lines.mfi[0] < self.lines.signal[0]:
#             return "下降"
#         else:
#             return "平稳"

#     def get_signal(self):
#         """根据MFI状态生成交易信号"""
#         # MFI从超卖区域向上突破
#         if self.lines.mfi[0] > self.p.oversold and self.lines.mfi[-1] <= self.p.oversold:
#             return "买入"

#         # MFI从超买区域向下突破
#         elif self.lines.mfi[0] < self.p.overbought and self.lines.mfi[-1] >= self.p.overbought:
#             return "卖出"

#         # MFI与信号线金叉
#         elif self.lines.mfi[0] > self.lines.signal[0] and self.lines.mfi[-1] <= self.lines.signal[-1]:
#             return "买入信号"

#         # MFI与信号线死叉
#         elif self.lines.mfi[0] < self.lines.signal[0] and self.lines.mfi[-1] >= self.lines.signal[-1]:
#             return "卖出信号"

#         # 检测背离
#         elif self.check_bullish_divergence():
#             return "看涨背离"

#         elif self.check_bearish_divergence():
#             return "看跌背离"

#         else:
#             return "无信号"

import backtrader as bt
import numpy as np


class MFI(bt.Indicator):
    """
    改进版 Money Flow Index (MFI) 指标
    
    计算流程：
      1. 计算典型价格：TP = (High + Low + Close) / 3
      2. 计算资金流：MF = TP * Volume
      3. 判断正负资金流：若TP大于前一期TP，则为正资金流，否则为负资金流
      4. 计算过去周期内正、负资金流的累加和，并计算资金流比率 MFR
      5. 计算 MFI = 100 - 100/(1 + MFR)
      
    改进点：
      1. 添加背离检测功能 - 识别价格与MFI的不一致走势
      2. 添加超买超卖区域定义 - 可自定义阈值
      3. 添加趋势状态指示 - 判断当前MFI的趋势状态
      4. 添加信号生成功能 - 基于MFI生成买入/卖出信号
    """
    lines = (
        'mfi',
        'signal',
    )
    params = (
        ('period', 14),  # MFI周期
        ('oversold', 20),  # 超卖阈值
        ('overbought', 80),  # 超买阈值
        ('signal_period', 9),  # 信号线周期
        ('divergence_period', 10),  # 背离检测周期
    )

    def __init__(self):
        # 计算典型价格
        tp = (self.data.high + self.data.low + self.data.close) / 3.0
        mf = tp * self.data.volume
        tp_prev = tp(-1)

        # 判断正负资金流（tp == tp_prev时，不做统计）
        pos_flow = bt.If(tp > tp_prev, mf, 0.0)
        neg_flow = bt.If(tp < tp_prev, mf, 0.0)

        # 累计正负资金流，避免除零
        pos_flow_sum = bt.indicators.SumN(pos_flow, period=self.p.period)
        neg_flow_sum = bt.indicators.SumN(neg_flow, period=self.p.period)

        # 计算资金流比率和MFI
        mfr = pos_flow_sum / (neg_flow_sum + 1e-10)
        self.lines.mfi = 100 - (100 / (1 + mfr))

        # 计算信号线 (MFI的移动平均)
        self.lines.signal = bt.indicators.EMA(self.lines.mfi,
                                              period=self.p.signal_period)

        # 记录最近的极值用于背离检测
        self.price_highs = []  # 价格高点列表 (bar_index, price)
        self.price_lows = []  # 价格低点列表 (bar_index, price)
        self.mfi_highs = []  # MFI高点列表 (bar_index, mfi_val)
        self.mfi_lows = []  # MFI低点列表 (bar_index, mfi_val)

        # 添加最小周期
        self.addminperiod(max(self.p.period, self.p.divergence_period) + 1)

    def next(self):
        # 在每个 bar 都检测是否出现新的高低点
        self.update_swing_points()

    def update_swing_points(self):
        """更新MFI和价格的极值点"""
        # 检查是否形成价格高点
        if self.is_high_point(self.data.close, self.p.divergence_period):
            self.price_highs.append((len(self), self.data.close[0]))
            self.mfi_highs.append((len(self), self.lines.mfi[0]))
            # 保留最近的3个高点
            if len(self.price_highs) > 3:
                self.price_highs.pop(0)
                self.mfi_highs.pop(0)

        # 检查是否形成价格低点
        if self.is_low_point(self.data.close, self.p.divergence_period):
            self.price_lows.append((len(self), self.data.close[0]))
            self.mfi_lows.append((len(self), self.lines.mfi[0]))
            # 保留最近的3个低点
            if len(self.price_lows) > 3:
                self.price_lows.pop(0)
                self.mfi_lows.pop(0)

    def is_high_point(self, series, lookback=5):
        """检查当前值（series[0]）是否在最近 lookback 根内最高"""
        if len(series) < lookback:
            return False
        current_val = series[0]
        for i in range(1, lookback):
            if series[-i] > current_val:
                return False
        return True

    def is_low_point(self, series, lookback=5):
        """检查当前值（series[0]）是否在最近 lookback 根内最低"""
        if len(series) < lookback:
            return False
        current_val = series[0]
        for i in range(1, lookback):
            if series[-i] < current_val:
                return False
        return True

    def check_bullish_divergence(self):
        """检查看涨背离：价格创新低，但MFI未创新低"""
        if len(self.price_lows) < 2 or len(self.mfi_lows) < 2:
            return False

        # 获取最近两个低点 (bar_index, val)
        (bar_curr_p, curr_price_low), (bar_prev_p,
                                       prev_price_low) = self.price_lows[-2:]
        (bar_curr_m, curr_mfi_low), (bar_prev_m,
                                     prev_mfi_low) = self.mfi_lows[-2:]

        # 判断是否为看涨背离
        return (curr_price_low < prev_price_low) and (curr_mfi_low
                                                      > prev_mfi_low)

    def check_bearish_divergence(self):
        """检查看跌背离：价格创新高，但MFI未创新高"""
        if len(self.price_highs) < 2 or len(self.mfi_highs) < 2:
            return False

        # 获取最近两个高点
        (bar_curr_p,
         curr_price_high), (bar_prev_p,
                            prev_price_high) = self.price_highs[-2:]
        (bar_curr_m, curr_mfi_high), (bar_prev_m,
                                      prev_mfi_high) = self.mfi_highs[-2:]

        # 判断是否为看跌背离
        return (curr_price_high > prev_price_high) and (curr_mfi_high
                                                        < prev_mfi_high)

    def get_zone(self):
        """获取当前MFI的区域状态"""
        if self.lines.mfi[0] > self.p.overbought:
            return "超买"
        elif self.lines.mfi[0] < self.p.oversold:
            return "超卖"
        else:
            return "中性"

    def get_trend(self):
        """获取当前MFI的趋势状态"""
        # 简单地使用MFI与其信号线的关系判断趋势
        if self.lines.mfi[0] > self.lines.signal[0]:
            return "上升"
        elif self.lines.mfi[0] < self.lines.signal[0]:
            return "下降"
        else:
            return "平稳"

    def get_signal(self):
        """
        根据MFI状态生成交易信号
        注意：若多个条件同时满足，只会返回最先匹配到的信号
        """
        # MFI从超卖区域向上突破
        if self.lines.mfi[0] > self.p.oversold and self.lines.mfi[
                -1] <= self.p.oversold:
            return "买入"

        # MFI从超买区域向下突破
        elif self.lines.mfi[0] < self.p.overbought and self.lines.mfi[
                -1] >= self.p.overbought:
            return "卖出"

        # MFI与信号线金叉
        elif (self.lines.mfi[0] > self.lines.signal[0]
              and self.lines.mfi[-1] <= self.lines.signal[-1]):
            return "买入信号"

        # MFI与信号线死叉
        elif (self.lines.mfi[0] < self.lines.signal[0]
              and self.lines.mfi[-1] >= self.lines.signal[-1]):
            return "卖出信号"

        # 检测背离
        elif self.check_bullish_divergence():
            return "看涨背离"

        elif self.check_bearish_divergence():
            return "看跌背离"

        else:
            return "无信号"
