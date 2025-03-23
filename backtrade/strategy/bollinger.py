import backtrader as bt


class BollingerStrategyEnhanced(bt.Strategy):
    """
    改良的布林带策略示例：
    1) 使用 bracket_order 创建“主订单 + 止盈 + 止损”，互为 OCO；
    2) 止盈止损基于 ATR 动态计算，避免固定百分比在不同波动阶段的不适用；
    3) 示例中简单加了一个中轨方向过滤（可选），减小假突破概率；
    4) 通过“单笔风险占总资金的固定比例”来动态计算 size，而非简单的 target_percent=0.9。
    """

    params = (
        # --- Bollinger Bands 参数 ---
        ('period', 20),
        ('devfactor', 2),
        # --- ATR 止盈止损参数 ---
        ('atr_period', 14),  # ATR周期
        ('atr_stop_loss', 1.5),  # 止损倍数
        ('atr_take_profit', 3.0),  # 止盈倍数
        # --- 资金管理 ---
        ('risk_per_trade', 0.01),  # 单笔风险占总资金的 1%
        # --- 中轨方向过滤（可关闭） ---
        ('use_mid_filter', True),  # 是否启用“价格在中轨上方才做多/下方才做空”的过滤
    )

    def log(self, txt, dt=None):
        """自定义日志函数，可在调试或回测时使用"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {txt}")

    def __init__(self):
        # === 收盘价引用 ===
        self.dataclose = self.datas[0].close

        # === 定义布林带指标 ===
        self.boll = bt.indicators.BollingerBands(self.dataclose,
                                                 period=self.p.period,
                                                 devfactor=self.p.devfactor)
        # 上轨: self.boll.top, 中轨: self.boll.mid, 下轨: self.boll.bot

        # === 定义 ATR 指标，用于动态止盈止损 ===
        self.atr = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)

        # === 跟踪当前挂单（如果有的话） ===
        self.order = None

    def notify_order(self, order):
        """
        订单状态更新回调。
        对于 bracket_order，会返回 3 个子订单：
          1) 主订单 (parent)
          2) 止损订单 (stop)
          3) 止盈订单 (limit)
        """
        if order.status in [order.Submitted, order.Accepted]:
            # 订单提交/接受后，不做特殊处理
            return

        # 订单完成
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

        # 订单取消/保证金不足/拒绝
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("[警告] 订单取消/保证金不足/拒绝")
            self.order = None

    def notify_trade(self, trade):
        """
        交易关闭时输出盈亏
        """
        if trade.isclosed:
            self.log(f"[交易结束] 毛收益: {trade.pnl:.2f}, 净收益: {trade.pnlcomm:.2f}")

    def next(self):
        """
        核心交易逻辑：
        1) 当还没有挂单或持仓时，依据布林带突破并回收的信号尝试开仓；
        2) 用 bracket_order 同时绑定止盈止损，互为 OCO。
        """
        if self.order:
            # 若有订单在途，则不重复下单
            return

        # 如果数据还不够长（如初始 warm-up 期间），直接跳过
        if len(self) < max(self.p.period, self.p.atr_period):
            return

        close_price = self.dataclose[0]
        top = self.boll.top[0]
        bot = self.boll.bot[0]
        mid = self.boll.mid[0]  # 中轨

        # =========== 【1】无持仓 -> 寻找开仓机会 ===========
        if not self.position:
            # --- 信号1：下轨突破买入 ---
            # (上一根 < 下轨) 且 (当前收盘 > 下轨)
            # + 可选过滤： close > 中轨则更倾向做多
            if (self.dataclose[-1] < self.boll.bot[-1]) and (close_price
                                                             > bot):
                if (not self.p.use_mid_filter) or (close_price > mid):
                    self.log(
                        f"[买入信号] 收盘价下轨突破 -> 准备开多, Close={close_price:.2f}")
                    self.buy_bracket_with_atr(is_long=True)

            # --- 信号2：上轨突破做空 ---
            # (上一根 > 上轨) 且 (当前收盘 < 上轨)
            # + 可选过滤： close < 中轨则更倾向做空
            elif (self.dataclose[-1] > self.boll.top[-1]) and (close_price
                                                               < top):
                if (not self.p.use_mid_filter) or (close_price < mid):
                    self.log(
                        f"[做空信号] 收盘价上轨突破 -> 准备开空, Close={close_price:.2f}")
                    self.buy_bracket_with_atr(is_long=False)
        else:
            # =========== 【2】已有持仓 -> 交给 bracket 止盈止损处理 ===========
            pass

    def buy_bracket_with_atr(self, is_long=True):
        """
        用 bracket_order 下单，并根据 ATR 动态计算止盈止损距离。
        示例：若想在行情波动较大时，自动加大止损和止盈距离。
        """
        close_price = self.dataclose[0]
        atr_value = self.atr[0]

        # 1) 先计算止损、止盈价
        stop_dist = self.p.atr_stop_loss * atr_value  # 距离=ATR倍数
        tp_dist = self.p.atr_take_profit * atr_value

        if is_long:
            # 多头
            entry_price = close_price
            stop_price = entry_price - stop_dist
            limit_price = entry_price + tp_dist
        else:
            # 空头
            entry_price = close_price
            stop_price = entry_price + stop_dist
            limit_price = entry_price - tp_dist

        # 2) 计算下单手数 size
        #    简单示例：当止损被打时，仅损失总资金的 self.p.risk_per_trade (例如 1%)
        #    -> (entry_price - stop_price) * size ≈ total_value * risk_per_trade
        #    对多头和空头分别做一个绝对值处理
        total_value = self.broker.getvalue()
        risk_amount = total_value * self.p.risk_per_trade  # 风险资金
        if is_long:
            risk_per_share = (entry_price - stop_price)
        else:
            risk_per_share = (stop_price - entry_price)

        # 避免出现 ATR 非常小或非常大的极端情况
        if risk_per_share <= 0:
            self.log("[警告] 风险距离 <= 0, 无法下单。检查 ATR 或价格逻辑。")
            return

        size = risk_amount / risk_per_share
        # 有些交易标的可能需要整数手数，或者加杠杆做调整，可在此做 round() 或其他计算
        size = int(size)  # 取整

        if size <= 0:
            self.log("[警告] 计算出的下单手数 <= 0, 跳过。")
            return

        # 3) 使用 bracket_order 下单
        if is_long:
            self.log(
                f"[提交买Bracket] Buy Price={entry_price:.2f}, Stop={stop_price:.2f}, TP={limit_price:.2f}, Size={size}"
            )
            self.order = self.buy_bracket(
                size=size,
                price=entry_price,  # 主订单价格(若使用限价，可指定limit价；这里用当前价格下市价单可写 None)
                stopprice=stop_price,  # 止损触发价
                limitprice=limit_price,  # 止盈价
                # exectype=bt.Order.Market  # 如果你要市价单，可以把主订单设成市价
            )
        else:
            self.log(
                f"[提交卖Bracket] Sell Price={entry_price:.2f}, Stop={stop_price:.2f}, TP={limit_price:.2f}, Size={size}"
            )
            self.order = self.sell_bracket(
                size=size,
                price=entry_price,
                stopprice=stop_price,
                limitprice=limit_price,
                # exectype=bt.Order.Market
            )

    def stop(self):
        """回测结束时输出最终市值"""
        self.log(f"[回测结束] 最终市值: {self.broker.getvalue():.2f}")
