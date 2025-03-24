import backtrader as bt
from .base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    """
    基于RSI指标的交易策略

    策略逻辑：
    1. 当RSI指标从超卖区域（低于设定的rsi_oversold）向上突破时，产生买入信号；
    2. 当RSI指标从超买区域（高于设定的rsi_overbought）向下突破时，产生卖出信号；
    3. 为防止频繁交易，信号之间需要满足一定的间隔（min_bars_between_signals）。
    
    参数说明：
      - rsi_period: RSI计算周期，默认14
      - rsi_oversold: 超卖阈值，默认30
      - rsi_overbought: 超买阈值，默认70
      - min_bars_between_signals: 信号间至少等待的Bar数量，默认3
      - log_level, collect_signals: 继承自BaseStrategy的日志及信号收集参数
    """
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('min_bars_between_signals', 3),
        ('log_level', BaseStrategy.LOG_LEVEL_INFO),
        ('collect_signals', True),
    )

    def __init__(self):
        super().__init__()
        # 初始化RSI指标
        self.rsi = bt.indicators.RSI(self.data.close,
                                     period=self.params.rsi_period)
        # 用于限制信号间隔，防止过于频繁的交易
        self.last_signal_bar = -self.params.min_bars_between_signals

    def next(self):
        super().next()  # 记录资产净值
        current_bar = len(self)

        # 如果距离上次信号未达到设定间隔，则不生成信号
        if current_bar - self.last_signal_bar < self.params.min_bars_between_signals:
            return

        # 没有持仓时检查买入信号
        if not self.position:
            # 当RSI从超卖区域向上突破（当前RSI > rsi_oversold且前一BarRSI<=rsi_oversold）时，买入
            if self.rsi[0] > self.params.rsi_oversold and self.rsi[
                    -1] <= self.params.rsi_oversold:
                self.execute_buy()
        else:
            # 持仓时检查卖出信号：RSI从超买区域向下突破（当前RSI < rsi_overbought且前一BarRSI>=rsi_overbought）时，卖出
            if self.rsi[0] < self.params.rsi_overbought and self.rsi[
                    -1] >= self.params.rsi_overbought:
                self.execute_sell()

    def execute_buy(self):
        """
        执行买入操作：计算可买数量并下买单，同时记录信号和日志
        """
        price = self.data.close[0]
        size = self.calc_max_shares(price)
        if size <= 0:
            return
        self.order = self.buy(size=size)
        self.last_signal_bar = len(self)
        self.log(f'RSI买入信号: 价格={price:.2f}, 数量={size}')

    def execute_sell(self):
        """
        执行卖出操作：全仓卖出，并记录信号和日志
        """
        if self.position:
            price = self.data.close[0]
            size = self.position.size
            self.order = self.sell(size=size)
            self.last_signal_bar = len(self)
            self.log(f'RSI卖出信号: 价格={price:.2f}, 数量={size}')


class NaiveRsiStrategy(bt.Strategy):
    """
    使用内置 RSI 指标的策略示例：
    - 仅当无持仓且 RSI 低于超卖阈值时，下满仓单；
    - 仅当有持仓且 RSI 高于超买阈值时，清仓；
    - 订单部分成交时取消剩余部分，防止订单一直挂单阻塞新信号；
    - 日志输出中文信息，包含时间戳和订单执行细节。
    """

    params = (
        ("period", 14),  # RSI 计算周期
        ("overbought", 70),  # 超买阈值
        ("oversold", 30),  # 超卖阈值
    )

    def log(self, txt, dt=None):
        """统一的日志输出函数，使用中文并打印具体时间。"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}  {txt}")

    def __init__(self):
        self.dataclose = self.datas[0].close
        # 使用 Backtrader 内置的 RSI 指标，仅基于最近 period 个周期的数据
        self.rsi = bt.indicators.RSI(self.data, period=self.params.period)
        self.order = None  # 用于跟踪当前活跃订单，避免重复下单

    def notify_order(self, order):
        """
        订单状态改变时调用：
        - 在订单提交/接收时打印日志；
        - 对于部分成交的订单，主动取消剩余部分，避免长时间挂单阻塞新交易信号；
        - 在订单完全成交或取消/拒绝时，重置 self.order。
        """
        if order.status in [order.Submitted, order.Accepted]:
            self.log(f"订单状态: {order.getstatusname()} (提交/接收)，等待成交...")
            return

        if order.status == order.Partial:
            self.log(
                f"订单部分成交: 剩余数量={order.created.size - order.executed.size}，已成交数量={order.executed.size}"
            )
            # 主动取消未成交部分，确保不会阻塞后续下单
            self.cancel(order)
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(
                    f"买单执行: 成交量={order.executed.size}, 成交价={order.executed.price:.2f}, "
                    f"订单金额={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}"
                )
            else:
                self.log(
                    f"卖单执行: 成交量={order.executed.size}, 成交价={order.executed.price:.2f}, "
                    f"订单金额={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}"
                )
            self.order = None

        elif order.status in [order.Canceled, order.Rejected]:
            self.log(f"订单取消/拒绝: {order.getstatusname()}")
            self.order = None

    def next(self):
        """
        每根Bar调用一次：
        - 如果已有订单挂单，则不执行新交易；
        - 根据当前 RSI 信号决定是否全仓买入或清仓。
        """
        dt = self.data.datetime.datetime(0)
        # 可选：打印当前资金、持仓和收盘价信息
        # cash = self.broker.getcash()
        # value = self.broker.getvalue()
        # pos_size = self.position.size
        # self.log(f"当前Bar={dt}, 收盘价={self.dataclose[0]:.2f}, 资金={cash:.2f}, 总市值={value:.2f}, 持仓数={pos_size}")

        # 如果已有挂单，直接返回
        if self.order:
            return

        current_rsi = self.rsi[0]

        # 无持仓时，RSI低于超卖阈值则满仓买入
        if not self.position:
            if current_rsi < self.params.oversold:
                self.log(
                    f"RSI={current_rsi:.2f} < 超卖阈值({self.params.oversold:.2f})，准备满仓买入，当前价格={self.dataclose[0]:.2f}"
                )
                self.order = self.order_target_percent(target=1.0)
        # 有持仓时，RSI高于超买阈值则清仓
        else:
            if current_rsi > self.params.overbought:
                self.log(
                    f"RSI={current_rsi:.2f} > 超买阈值({self.params.overbought:.2f})，准备清仓，当前价格={self.dataclose[0]:.2f}"
                )
                self.order = self.order_target_percent(target=0.0)

    def stop(self):
        """回测结束时输出最终市值。"""
        self.log(f"回测结束 - 最终总市值: {self.broker.getvalue():.2f}")
