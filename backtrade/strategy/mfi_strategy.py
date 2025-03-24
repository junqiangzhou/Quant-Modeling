import backtrader as bt

from backtrade.indicator.mfi import MFI
from backtrade.strategy.base_strategy import BaseStrategy


class MFIStrategy(BaseStrategy):
    """
    改进后的MFI背离策略主要优化点：
    1. 动态止盈止损机制
    2. 极值检测灵敏度提升
    3. 增加趋势过滤
    4. 优化信号间隔逻辑
    """
    params = (
        ('mfi_period', 14),
        ('mfi_oversold', 25),  # 放宽超卖阈值
        ('mfi_overbought', 75),  # 放宽超买阈值
        ('divergence_lookback', 50),  # 延长背离检测窗口
        ('swing_lookback', 3),  # 缩短极值检测窗口
        ('stop_loss', 0.03),  # 收紧止损
        ('take_profit', 0.15),  # 扩大止盈
        ('trailing_stop', 0.03),  # 新增跟踪止损比例
        ('min_bars_between_signals', 3),  # 缩短信号间隔
        ('use_trend_filter', True),  # 启用趋势过滤
        ('log_level', BaseStrategy.LOG_LEVEL_INFO),
        ('collect_signals', True),
    )

    def __init__(self):
        super().__init__()

        # 核心指标
        self.mfi = MFI(self.data,
                       period=self.params.mfi_period,
                       oversold=self.params.mfi_oversold,
                       overbought=self.params.mfi_overbought)

        # 高低点指示器（用于辅助判断价格是否处于极值）
        self.swing_low = bt.indicators.Lowest(
            self.data.close, period=self.params.swing_lookback + 1)
        self.swing_high = bt.indicators.Highest(
            self.data.close, period=self.params.swing_lookback + 1)

        # 趋势过滤指标
        if self.params.use_trend_filter:
            self.ma200 = bt.indicators.SMA(self.data.close, period=200)
            self.ma50 = bt.indicators.SMA(self.data.close, period=50)

        # 维护价格和 MFI 的极值点
        self.price_lows = []
        self.price_highs = []
        self.mfi_lows = []
        self.mfi_highs = []

        # 状态变量
        self.peak_price = None  # 跟踪止损
        self.order = None
        self.last_signal_bar = -self.params.min_bars_between_signals
        self.buy_price = None

    def next(self):
        super().next()  # 记录资产净值
        current_bar = len(self)

        # 每个 bar 更新价格 & MFI 的极值
        self.update_swings()

        # 如果已有挂单，先不执行
        if self.order:
            return

        # 先检查是否需要止盈止损（如果已经有持仓）
        if self.position:
            if self.check_stop_conditions():
                self.execute_sell()
                return

        # 检查信号间隔
        if current_bar - self.last_signal_bar < self.params.min_bars_between_signals:
            return

        # 主多头逻辑
        if not self.position:
            if self.check_entry_signals():
                self.execute_buy()
        else:
            # 已持仓，看是否满足平仓信号
            if self.check_exit_signals():
                self.execute_sell()
            else:
                # 更新跟踪止损
                self.update_trailing_stop()

    def update_swings(self):
        """价格极值 + MFI 极值检测"""
        # 价格低点且 MFI 接近或处于超卖
        if self.data.close[0] == self.swing_low[0] and self.mfi[0] < (
                self.params.mfi_oversold + 5):
            self._add_swing_point(self.price_lows, self.mfi_lows)

        # 价格高点且 MFI 接近或处于超买
        if self.data.close[0] == self.swing_high[0] and self.mfi[0] > (
                self.params.mfi_overbought - 5):
            self._add_swing_point(self.price_highs, self.mfi_highs)

    def _add_swing_point(self, price_points, mfi_points):
        bar_index = len(self)
        price_points.append((bar_index, self.data.close[0]))
        mfi_points.append((bar_index, self.mfi[0]))
        # 只保留最近的 3 次极值
        if len(price_points) > 3:
            price_points.pop(0)
            mfi_points.pop(0)

    def check_bullish_divergence(self):
        """看涨背离：价格创新低，MFI 没创新低"""
        if len(self.price_lows) < 2:
            return False

        (curr_bar_p, curr_price), (prev_bar_p,
                                   prev_price) = self.price_lows[-2:]
        (curr_bar_m, curr_mfi), (prev_bar_m, prev_mfi) = self.mfi_lows[-2:]

        # 时间间隔要在 divergence_lookback 以内
        if curr_bar_p - prev_bar_p > self.params.divergence_lookback:
            return False

        # 价格新低 & MFI 抬高
        return (curr_price < prev_price) and (curr_mfi > prev_mfi)

    def check_bearish_divergence(self):
        """看跌背离：价格创新高，MFI 没创新高"""
        if len(self.price_highs) < 2:
            return False

        (curr_bar_p, curr_price), (prev_bar_p,
                                   prev_price) = self.price_highs[-2:]
        (curr_bar_m, curr_mfi), (prev_bar_m, prev_mfi) = self.mfi_highs[-2:]

        if curr_bar_p - prev_bar_p > self.params.divergence_lookback:
            return False

        return (curr_price > prev_price) and (curr_mfi < prev_mfi)

    def check_entry_signals(self):
        """多头入场条件"""
        # 趋势过滤
        if self.params.use_trend_filter:
            # 简单示例：价格大于 MA200 且 MA50 > MA200
            if self.data.close[0] < self.ma200[0] or self.ma50[0] < self.ma200[
                    0]:
                return False

        # MFI从超卖区域突破
        mfi_cross_up = (self.mfi[0] > self.params.mfi_oversold
                        and self.mfi[-1] <= self.params.mfi_oversold)

        # 是否看涨背离
        bullish_div = self.check_bullish_divergence()

        return mfi_cross_up and bullish_div

    def check_exit_signals(self):
        """多头出场条件"""
        # MFI从超买区域回落
        mfi_cross_down = (self.mfi[0] < self.params.mfi_overbought
                          and self.mfi[-1] >= self.params.mfi_overbought)

        # 看跌背离
        bearish_div = self.check_bearish_divergence()

        return mfi_cross_down or bearish_div

    def update_trailing_stop(self):
        """动态更新止损（若有需要）"""
        # 若当前价格创新高，则更新 peak_price
        if self.peak_price is None or self.data.high[0] > self.peak_price:
            self.peak_price = self.data.high[0]

    def check_stop_conditions(self):
        """硬止损 & 跟踪止损 & 止盈"""
        if not self.position:
            return False
        current_price = self.data.close[0]

        # 硬止损
        if current_price <= (self.buy_price * (1 - self.params.stop_loss)):
            self.log(f'触发硬止损@ {current_price:.2f}')
            return True

        # 止盈
        if current_price >= (self.buy_price * (1 + self.params.take_profit)):
            self.log(f'触发止盈@ {current_price:.2f}')
            return True

        # 跟踪止损
        if self.peak_price and current_price <= (
                self.peak_price * (1 - self.params.trailing_stop)):
            self.log(f'触发跟踪止损@ {current_price:.2f}')
            return True

        return False

    def execute_buy(self):
        """执行买入"""
        price = self.data.close[0]
        size = self.calc_max_shares(price)
        if size <= 0:
            return

        self.buy_price = price
        self.peak_price = price
        self.order = self.buy(size=size)
        self.last_signal_bar = len(self)
        self.log(f'执行买入: 价格={price:.2f}, 数量={size}')

    def execute_sell(self):
        """执行卖出"""
        if self.position:
            price = self.data.close[0]
            size = self.position.size
            self.order = self.sell(size=size)
            self.last_signal_bar = len(self)
            self.log(f'执行卖出: 价格={price:.2f}, 数量={size}')
