# import backtrader as bt
# import numpy as np

# class OnBalanceVolume(bt.Indicator):
#     """
#     改进版 On-Balance Volume (OBV) 指标

#     计算公式:
#     OBV(i) = OBV(i-1) + volume(i),   if close(i) > close(i-1)
#            = OBV(i-1) - volume(i),   if close(i) < close(i-1)
#            = OBV(i-1),              otherwise

#     改进点:
#     1. 支持归一化处理 - 使OBV值更易于比较和可视化
#     2. 支持基于成交量大小的有效性过滤 - 在成交量极小时OBV可能失真
#     3. 支持平滑处理 - 通过EMA使OBV曲线更平滑
#     4. 添加信号线 - 便于识别趋势变化
#     """

#     # 定义输出线（Backtrader中的指标必须至少定义一条 lines）
#     lines = ('obv', 'obv_ema', 'obv_signal',)

#     # 扩展参数
#     params = (
#         ('use_volume_at_first_bar', False),  # 是否从 volume(0) 开始累计
#         ('normalize', False),                # 是否归一化OBV值
#         ('normalize_window', 100),           # 归一化窗口长度
#         ('vol_min_pct', 0.2),                # 最小有效成交量百分比
#         ('smooth_period', 5),                # OBV平滑的EMA周期
#         ('signal_period', 20),               # 信号线EMA周期
#     )

#     def __init__(self):
#         # 为了确保我们可以访问到 data.close[-1]，必须要在 init 里指定最小周期
#         self.addminperiod(max(2, self.p.normalize_window))

#         # 初始化成交量移动平均线用于有效性过滤
#         self.vol_ma = bt.indicators.SimpleMovingAverage(self.data.volume, period=20)

#         # OBV平滑处理
#         self.obv_smooth = bt.indicators.EMA(self.lines.obv, period=self.p.smooth_period)

#         # OBV信号线
#         self.signal_line = bt.indicators.EMA(self.lines.obv, period=self.p.signal_period)

#         # 将平滑后的OBV和信号线绑定到line上
#         self.lines.obv_ema = self.obv_smooth
#         self.lines.obv_signal = self.signal_line

#     def nextstart(self):
#         """
#         初始化 OBV 的起始值
#         """
#         # 让 OBV(0) = 0（或初始 volume），以下示例以0为起点
#         if self.p.use_volume_at_first_bar:
#             self.lines.obv[0] = self.data.volume[0]
#         else:
#             self.lines.obv[0] = 0

#         # nextstart()执行完后，后续计算会在 next() 中
#         super().nextstart()

#     def next(self):
#         """
#         计算OBV值，增加归一化和有效性过滤
#         """
#         # 获取当前和前一个收盘价
#         current_close = self.data.close[0]
#         prev_close = self.data.close[-1]

#         # 获取当前成交量及其有效性
#         current_volume = self.data.volume[0]
#         volume_valid = current_volume >= self.vol_ma[0] * self.p.vol_min_pct

#         # 根据价格变动更新OBV值
#         if not volume_valid:
#             # 成交量过小，OBV保持不变
#             self.lines.obv[0] = self.lines.obv[-1]
#         elif current_close > prev_close:
#             self.lines.obv[0] = self.lines.obv[-1] + current_volume
#         elif current_close < prev_close:
#             self.lines.obv[0] = self.lines.obv[-1] - current_volume
#         else:
#             # 收盘价相同，OBV 不变
#             self.lines.obv[0] = self.lines.obv[-1]

#         # 如果启用归一化，则对OBV值进行归一化处理
#         if self.p.normalize and len(self) >= self.p.normalize_window:
#             # 提取归一化窗口内的OBV值
#             obv_series = np.array([self.lines.obv[-i] for i in range(self.p.normalize_window)])

#             # 计算最大最小值
#             obv_min = np.min(obv_series)
#             obv_max = np.max(obv_series)
#             obv_range = obv_max - obv_min

#             # 避免除以零
#             if obv_range > 0:
#                 # 归一化到0-100范围
#                 self.lines.obv[0] = 100 * (self.lines.obv[0] - obv_min) / obv_range

#     def is_diverging(self, price_direction):
#         """
#         检测OBV与价格是否出现背离

#         参数:
#         price_direction - 价格方向，1表示上涨，-1表示下跌

#         返回:
#         True如果检测到背离，否则False
#         """
#         obv_direction = 1 if self.lines.obv[0] > self.lines.obv[-5] else -1
#         return price_direction != obv_direction

import backtrader as bt
import numpy as np


class OnBalanceVolume(bt.Indicator):
    """
    改进版 On-Balance Volume (OBV) 指标
    
    计算公式:
    OBV(i) = OBV(i-1) + volume(i),   if close(i) > close(i-1)
           = OBV(i-1) - volume(i),   if close(i) < close(i-1)
           = OBV(i-1),              otherwise
    
    改进点:
    1. 支持归一化处理 - 使OBV值更易于比较和可视化
    2. 支持基于成交量大小的有效性过滤 - 在成交量极小时OBV可能失真
    3. 支持平滑处理 - 通过EMA使OBV曲线更平滑
    4. 添加信号线 - 便于识别趋势变化
    """

    lines = (
        'obv',
        'obv_ema',
        'obv_signal',
    )

    params = (
        ('use_volume_at_first_bar', False),  # 是否从 volume(0) 开始累计
        ('normalize', False),  # 是否归一化OBV值
        ('normalize_window', 100),  # 归一化窗口长度
        ('vol_min_pct', 0.2),  # 最小有效成交量百分比
        ('smooth_period', 5),  # OBV平滑的EMA周期
        ('signal_period', 20),  # 信号线EMA周期
    )

    def __init__(self):
        self.addminperiod(max(2, self.p.normalize_window))

        # 成交量移动平均线用于有效性过滤
        self.vol_ma = bt.indicators.SimpleMovingAverage(self.data.volume,
                                                        period=20)

        # OBV平滑处理
        self.obv_smooth = bt.indicators.EMA(self.lines.obv,
                                            period=self.p.smooth_period)

        # OBV信号线
        self.signal_line = bt.indicators.EMA(self.lines.obv,
                                             period=self.p.signal_period)

        # 将平滑后的OBV和信号线绑定到 lines
        self.lines.obv_ema = self.obv_smooth
        self.lines.obv_signal = self.signal_line

    def nextstart(self):
        """
        初始化 OBV 的起始值
        """
        if self.p.use_volume_at_first_bar:
            self.lines.obv[0] = self.data.volume[0]
        else:
            self.lines.obv[0] = 0

        super().nextstart()

    def next(self):
        # 获取当前/前一个收盘价
        current_close = self.data.close[0]
        prev_close = self.data.close[-1]

        # 获取当前成交量及其有效性
        current_volume = self.data.volume[0]
        volume_valid = (current_volume >= self.vol_ma[0] * self.p.vol_min_pct)

        # 根据价格变动更新OBV值
        if not volume_valid:
            # 成交量过小，OBV保持不变
            self.lines.obv[0] = self.lines.obv[-1]
        elif current_close > prev_close:
            self.lines.obv[0] = self.lines.obv[-1] + current_volume
        elif current_close < prev_close:
            self.lines.obv[0] = self.lines.obv[-1] - current_volume
        else:
            self.lines.obv[0] = self.lines.obv[-1]

        # 如果启用归一化，则对OBV值进行归一化
        if self.p.normalize and len(self) >= self.p.normalize_window:
            obv_series = np.array(
                [self.lines.obv[-i] for i in range(self.p.normalize_window)])
            obv_min = np.min(obv_series)
            obv_max = np.max(obv_series)
            obv_range = obv_max - obv_min
            if obv_range > 0:
                self.lines.obv[0] = 100 * (self.lines.obv[0] -
                                           obv_min) / obv_range

    def is_diverging(self, price_direction):
        """
        检测OBV与价格是否出现背离（简单示例）
        
        参数:
        price_direction - 价格方向，1表示上涨，-1表示下跌
        
        返回:
        True如果检测到背离，否则False
        """
        obv_direction = 1 if self.lines.obv[0] > self.lines.obv[-5] else -1
        return price_direction != obv_direction
