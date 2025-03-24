import backtrader as bt


class MACrossoverStrategy(bt.Strategy):
    """
    MA Crossover 策略示例：
    - 当短周期均线上穿长周期均线时做多；
    - 当短周期均线下穿长周期均线时做空；
    - 开仓后自动设置固定止盈和固定止损单（可选）。
    
    参数：
    - ma_short_period: 短周期均线周期
    - ma_long_period:  长周期均线周期
    - target_pct:       每次开仓的目标资金占比
    - stop_loss:        固定止损百分比（对开仓价）
    - take_profit:      固定止盈百分比（对开仓价）
    """

    params = (
        ('ma_short_period', 5),
        ('ma_long_period', 15),
        ('target_pct', 0.9),
        ('stop_loss', 0.02),  # 2% 止损
        ('take_profit', 0.05),  # 5% 止盈
    )

    def log(self, txt, dt=None):
        """自定义日志函数，可在 debug 或回测时使用"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {txt}")

    def __init__(self):
        # ========== 1. 保存引用 ==========
        self.dataclose = self.datas[0].close

        # ========== 2. 跟踪订单和止盈止损单 ==========
        self.order = None  # 主订单
        self.stop_order = None  # 止损订单
        self.takeprofit_order = None  # 止盈订单

        # ========== 3. 定义均线、均线交叉指标 ==========
        self.ma_short = bt.indicators.SMA(self.dataclose,
                                          period=self.p.ma_short_period)
        self.ma_long = bt.indicators.SMA(self.dataclose,
                                         period=self.p.ma_long_period)
        # CrossOver 指标：大于0表示上穿，小于0表示下穿
        self.crossover = bt.indicators.CrossOver(self.ma_short, self.ma_long)

    def notify_order(self, order):
        """
        订单状态更新回调。
        这里需要注意区分：主订单、止盈单、止损单。
        """
        if order.status in [order.Submitted, order.Accepted]:
            # 订单提交/接受后不做特别处理
            return

        if order.status in [order.Completed]:
            # --- 主订单成交逻辑 ---
            if order == self.order:
                if order.isbuy():
                    self.log(
                        f"[成交] 买单执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                    )
                    # 买单完成后，为多头设置固定止盈止损
                    entry_price = order.executed.price
                    stop_price = entry_price * (1.0 - self.p.stop_loss)
                    tp_price = entry_price * (1.0 + self.p.take_profit)

                    # 止损单 (Stop)
                    self.stop_order = self.sell(exectype=bt.Order.Stop,
                                                price=stop_price)
                    self.log(f"[止损单提交] 多头止损价={stop_price:.2f}")

                    # 止盈单 (Limit)
                    self.takeprofit_order = self.sell(exectype=bt.Order.Limit,
                                                      price=tp_price)
                    self.log(f"[止盈单提交] 多头止盈价={tp_price:.2f}")

                elif order.issell():
                    self.log(
                        f"[成交] 卖单执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                    )
                    # 卖单完成后，为空头设置固定止盈止损
                    entry_price = order.executed.price
                    stop_price = entry_price * (1.0 + self.p.stop_loss)
                    tp_price = entry_price * (1.0 - self.p.take_profit)

                    # 止损单 (Stop)
                    self.stop_order = self.buy(exectype=bt.Order.Stop,
                                               price=stop_price)
                    self.log(f"[止损单提交] 空头止损价={stop_price:.2f}")

                    # 止盈单 (Limit)
                    self.takeprofit_order = self.buy(exectype=bt.Order.Limit,
                                                     price=tp_price)
                    self.log(f"[止盈单提交] 空头止盈价={tp_price:.2f}")

            # 若是止盈/止损单成交，也记录一下
            if order == self.stop_order:
                self.log(f"[触发止损] 价格={order.executed.price:.2f}")
                self.stop_order = None
                # 止损发生后，止盈单需要取消，避免持仓不一致
                if self.takeprofit_order:
                    self.cancel(self.takeprofit_order)
                    self.takeprofit_order = None

            if order == self.takeprofit_order:
                self.log(f"[触发止盈] 价格={order.executed.price:.2f}")
                self.takeprofit_order = None
                # 止盈发生后，止损单需要取消
                if self.stop_order:
                    self.cancel(self.stop_order)
                    self.stop_order = None

            self.order = None  # 主订单重置

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("[警告] 订单取消/保证金不足/拒绝")
            # 如果是主订单被拒绝或取消，也要重置
            if order == self.order:
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
        # 如果有订单在处理中，则不再下单
        if self.order:
            return

        # 如果已经有持仓，看一下是否需要在交叉反转时平仓或反向
        if self.position:
            # 多头持仓 && 均线出现死叉（短下穿长）
            if self.position.size > 0 and self.crossover[0] < 0:
                self.log("[平仓信号] 均线死叉，多头离场")
                # 先平掉当前多头仓位
                self.order = self.close()
                # 取消原有多头止盈止损单
                if self.stop_order:
                    self.cancel(self.stop_order)
                    self.stop_order = None
                if self.takeprofit_order:
                    self.cancel(self.takeprofit_order)
                    self.takeprofit_order = None

                # 选择是否反手做空：若需要可在此下单：
                # self.order = self.order_target_percent(target=-self.p.target_pct)

            # 空头持仓 && 均线出现金叉（短上穿长）
            elif self.position.size < 0 and self.crossover[0] > 0:
                self.log("[平仓信号] 均线金叉，空头离场")
                # 平掉当前空头仓位
                self.order = self.close()
                # 取消原有空头止盈止损单
                if self.stop_order:
                    self.cancel(self.stop_order)
                    self.stop_order = None
                if self.takeprofit_order:
                    self.cancel(self.takeprofit_order)
                    self.takeprofit_order = None

                # 选择是否反手做多：若需要可在此下单：
                # self.order = self.order_target_percent(target=self.p.target_pct)

        else:
            # 当前无持仓 => 根据均线交叉决定开仓方向
            if self.crossover[0] > 0:
                # 短周期上穿长周期 => 买入开多
                self.log("[做多信号] 均线金叉, 准备开多")
                self.order = self.order_target_percent(
                    target=self.p.target_pct)

            elif self.crossover[0] < 0:
                # 短周期下穿长周期 => 卖出做空
                self.log("[做空信号] 均线死叉, 准备开空")
                self.order = self.order_target_percent(
                    target=-self.p.target_pct)

    def stop(self):
        """回测结束时输出最终市值"""
        self.log(f"[回测结束] 最终市值: {self.broker.getvalue():.2f}")
