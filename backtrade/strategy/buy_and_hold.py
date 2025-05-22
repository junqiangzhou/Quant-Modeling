import backtrader as bt


class BuyAndHoldStrategy(bt.Strategy):
    """
    简单的买入并持有策略：
    1) 在回测开始时买入指定比例的资产
    2) 持有直到回测结束
    3) 可选择是否使用风险管理来计算头寸大小
    
    用作基准策略，比较其他策略的表现
    """

    params = (
        # --- 资金管理 ---
        ('target_percent', 0.95),  # 默认使用95%的资金买入
        # --- 风险管理（可选） ---
        ('use_risk_sizing', False),  # 是否使用风险管理计算头寸
        ('risk_per_trade', 0.01),  # 单笔风险占总资金的1%
        ('atr_period', 14),  # ATR周期
        ('atr_risk_factor', 2.0),  # ATR风险系数（确定止损距离）
        ('debug_mode', False))

    def log(self, txt, dt=None):
        """自定义日志函数，可在调试或回测时使用"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {txt}")

    def __init__(self):
        # 收盘价引用
        self.dataclose = self.datas[0].close

        # 跟踪当前挂单（如果有的话）
        self.order = None

        # 如果使用风险管理，创建ATR指标
        if self.p.use_risk_sizing:
            self.atr = bt.indicators.ATR(self.datas[0],
                                         period=self.p.atr_period)

        self.debug_mode = self.p.debug_mode

    def notify_order(self, order):
        """订单状态更新回调"""
        if order.status in [order.Submitted, order.Accepted]:
            # 订单提交/接受后，不做特殊处理
            return

        # 订单完成
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.debug_mode:
                    self.log(
                        f"[成交] 买单执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                    )
            elif order.issell():
                if self.debug_mode:
                    self.log(
                        f"[成交] 卖单执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                    )

            self.order = None

        # 订单取消/保证金不足/拒绝
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.debug_mode:
                self.log("[警告] 订单取消/保证金不足/拒绝")
            self.order = None

    def notify_trade(self, trade):
        """交易关闭时输出盈亏"""
        if trade.isclosed:
            if self.debug_mode:
                self.log(
                    f"[交易结束] 毛收益: {trade.pnl:.2f}, 净收益: {trade.pnlcomm:.2f}")

    def next(self):
        """
        策略核心逻辑：
        1) 在启动时买入目标仓位
        2) 之后不再做任何操作，持有至回测结束
        """
        # 如果有未完成订单，等待完成
        if self.order:
            return

        # 如果还没有持仓，执行买入
        if not self.position:
            # 等待数据预热完成（针对ATR情况）
            if self.p.use_risk_sizing and len(self) < self.p.atr_period:
                return

            self.buy_with_sizing()

    def buy_with_sizing(self):
        """根据策略参数选择合适的仓位管理方式进行买入"""
        close_price = self.dataclose[0]
        total_value = self.broker.getvalue()

        if self.p.use_risk_sizing:
            # === 基于风险的头寸管理 ===
            atr_value = self.atr[0]

            # 计算止损距离（仅用于头寸计算，实际不会设置止损单）
            stop_dist = self.p.atr_risk_factor * atr_value
            stop_price = close_price - stop_dist

            # 计算可承受的最大风险金额
            risk_amount = total_value * self.p.risk_per_trade

            # 计算头寸大小
            risk_per_share = close_price - stop_price

            # 安全检查
            if risk_per_share <= 0:
                if self.debug_mode:
                    self.log("[警告] 风险距离计算有误，使用目标百分比代替")
                size = int((total_value * self.p.target_percent) / close_price)
            else:
                size = int(risk_amount / risk_per_share)

                # 设置最大百分比限制，避免过度杠杆
                max_size = int(
                    (total_value * self.p.target_percent) / close_price)
                size = min(size, max_size)
        else:
            # === 简单的目标百分比头寸 ===
            size = int((total_value * self.p.target_percent) / close_price)

        # 确保至少买入1股
        size = max(1, size)

        # 执行买入
        if self.debug_mode:
            self.log(f"[买入] 执行买入并持有策略: 价格={close_price:.2f}, 数量={size}")
        self.order = self.buy(size=size)

    def stop(self):
        """回测结束时输出最终市值"""
        portfolio_value = self.broker.getvalue()
        if self.debug_mode:
            self.log(f"[回测结束] Buy & Hold 策略最终市值: {portfolio_value:.2f}")

        # 计算总收益率
        starting_value = self.broker.startingcash
        roi = (portfolio_value / starting_value - 1.0) * 100
        if self.debug_mode:
            self.log(f"[回测结束] 总收益率: {roi:.2f}%")
