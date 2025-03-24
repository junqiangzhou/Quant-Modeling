import backtrader as bt


class BollingerRSIStrategyV2(bt.Strategy):
    """
    优化版 RSI + 布林带策略示例，增加了“突破”判断和可选的 ATR 止损。

    参数：
    - rsi_period:      RSI 计算周期
    - rsi_overbought:  RSI 超买阈值
    - rsi_oversold:    RSI 超卖阈值
    - bb_period:       布林带周期
    - bb_devfactor:    布林带标准差倍数
    - target_pct:      每次开仓占用资金的目标比例
    - atr_period:      ATR 指标周期，用于止损
    - atr_stop_mult:   ATR 止损倍数（开仓后：止损价 = 进场价格 - atr_stop_mult * ATR）
    """
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('bb_period', 20),
        ('bb_devfactor', 2),
        ('target_pct', 0.9),
        # 新增 ATR 止损相关参数
        ('atr_period', 14),
        ('atr_stop_mult', 2.0),
    )

    def log(self, txt, dt=None):
        """自定义日志函数，可在 debug 或回测时使用"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {txt}")

    def __init__(self):
        # 保存收盘价引用
        self.dataclose = self.datas[0].close

        # 跟踪未完成订单（市价单）
        self.order = None
        # 跟踪止损订单
        self.stop_order = None

        # 指标：RSI、布林带、ATR
        self.rsi = bt.indicators.RSI(self.dataclose, period=self.p.rsi_period)
        self.bb = bt.indicators.BollingerBands(self.dataclose,
                                               period=self.p.bb_period,
                                               devfactor=self.p.bb_devfactor)
        self.atr = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)

    def notify_order(self, order):
        """
        订单状态更新回调。
        注意：在这里也需要处理止损单 StopOrder 的状态，例如被触发后记录日志。
        """
        if order.status in [order.Submitted, order.Accepted]:
            # 订单提交/接受后不做特别处理
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"[成交] 买单执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                )
            elif order.issell():
                self.log(
                    f"[成交] 卖单执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                )
            self.order = None

        # 若订单被取消、保证金不足、或被拒绝
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("[警告] 订单取消/保证金不足/拒绝")
            self.order = None

    def notify_trade(self, trade):
        """
        交易（trade）状态更新回调。
        一个 trade 可能包含多个 order。这里可以做更多统计或日志。
        """
        if not trade.isclosed:
            return
        # 交易关闭时输出盈亏
        self.log(f"[交易结束] 毛收益: {trade.pnl:.2f}, 净收益: {trade.pnlcomm:.2f}")

    def next(self):
        # 如果有挂单正在执行，直接返回（避免重复下单）
        if self.order:
            return

        # 获取当前收盘价、布林带上下轨、RSI 值
        close_price = self.dataclose[0]
        bb_lower = self.bb.lines.bot[0]
        bb_upper = self.bb.lines.top[0]
        rsi_value = self.rsi[0]

        # =============== 买入逻辑 ===============
        if not self.position:  # 无持仓
            # 条件1：上一根K线收盘价 < 下轨, 当前K线收盘价 > 下轨(即突破下轨)
            # 条件2：RSI 处于超卖区间
            if (self.dataclose[-1] < self.bb.lines.bot[-1] and close_price
                    > bb_lower) and (rsi_value < self.p.rsi_oversold):
                self.log(f"[买入信号] 收盘价由下轨下方向上突破下轨, RSI={rsi_value:.2f}")
                # 发出目标仓位订单（相当于市价买入到 p.target_pct 的仓位）
                self.order = self.order_target_percent(
                    target=self.p.target_pct)

                # 可选：开仓后下发止损单
                # 例如止损位设置为：当前价格 - ATR倍数
                stop_price = close_price - self.p.atr_stop_mult * self.atr[0]
                # 注意要在下单执行后再放止损, 这里演示用 next() 直接下单可能导致顺序竞争
                # 方式一: 简化处理，假设市价单很快完成，立即放 stop单
                if self.stop_order:
                    self.cancel(self.stop_order)
                self.stop_order = self.sell(exectype=bt.Order.Stop,
                                            price=stop_price)
                self.log(f"[止损单提交] 止损价={stop_price:.2f}")
        else:
            # =============== 卖出逻辑 ===============
            # 条件1：上一根K线收盘价 > 上轨, 当前K线收盘价 < 上轨(即从上轨上方向下突破)
            # 条件2：RSI 处于超买区间
            if (self.dataclose[-1] > self.bb.lines.top[-1] and close_price
                    < bb_upper) and (rsi_value > self.p.rsi_overbought):
                self.log(f"[卖出信号] 收盘价由上轨上方向下跌破上轨, RSI={rsi_value:.2f}")
                # 清空仓位
                self.order = self.order_target_percent(target=0.0)

                # 如果之前有止损单，则需要取消止损单，防止重复卖出
                if self.stop_order:
                    self.cancel(self.stop_order)
                    self.stop_order = None

    def stop(self):
        """回测结束时输出最终市值"""
        self.log(f"[回测结束] 最终市值: {self.broker.getvalue():.2f}")
