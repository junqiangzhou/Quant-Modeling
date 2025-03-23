import backtrader as bt
import math
from datetime import datetime, timedelta


class DoubleMAStrategy(bt.Strategy):
    """
    改良版双均线交叉策略：
    1) 当短周期均线上穿长周期均线 -> 做多；下穿 -> 做空（可选，若不想做空可去掉相关逻辑）。
    2) 利用 bracket_order + ATR 实现动态止损止盈，减少固定百分比在不同波动下的不适用。
    3) 通过单笔风险占总资金固定比例 (risk_per_trade) 来计算下单手数，兼顾资金管理。
    4) 附带简单的日志和交易结束信息输出，用于调试和验证回测结果。
    """

    params = (
        # --- 均线参数 ---
        ('fast_period', 20),
        ('slow_period', 50),
        # --- ATR 止盈止损参数 ---
        ('atr_period', 14),
        ('atr_stop_loss', 1.5),
        ('atr_take_profit', 3.0),
        # --- 资金管理 ---
        ('risk_per_trade', 0.01),  # 单笔风险占总资金的1%
        # --- 是否允许做空 ---
        ('allow_short', True),
    )

    def log(self, txt, dt=None):
        """自定义日志函数，可在调试或回测时使用"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {txt}")

    def __init__(self):
        # === 收盘价引用 ===
        self.dataclose = self.datas[0].close

        # === 短周期&长周期均线 ===
        self.ma_fast = bt.indicators.SMA(self.dataclose,
                                         period=self.p.fast_period)
        self.ma_slow = bt.indicators.SMA(self.dataclose,
                                         period=self.p.slow_period)

        # === 均线差值，用于判断金叉/死叉 ===
        # crossover > 0 表示短均线由下往上穿越长均线；< 0 表示由上往下穿越。
        self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)

        # === ATR 用于动态止损止盈 ===
        self.atr = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)

        # === 跟踪当前挂单（如果有的话） ===
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
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

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("[警告] 订单取消/保证金不足/拒绝")
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"[交易结束] 毛收益: {trade.pnl:.2f}, 净收益: {trade.pnlcomm:.2f}")

    def next(self):
        # 如果已有订单在路上，就不重复下单
        if self.order:
            return

        # 若数据尚未满足最小周期需求，直接跳过
        if len(self) < max(self.p.fast_period, self.p.slow_period,
                           self.p.atr_period):
            return

        # 检查金叉/死叉
        cross_value = self.crossover[0]
        close_price = self.dataclose[0]

        if not self.position:
            # ============ 没有持仓 -> 判断金叉做多 / 死叉做空 ============

            # 金叉: 短上穿长
            if cross_value > 0:
                self.log(f"[买入信号] 短均线 上穿 长均线 -> 准备开多, Close={close_price:.2f}")
                self.buy_bracket_with_atr(is_long=True)

            # 死叉: 短下穿长
            elif cross_value < 0 and self.p.allow_short:
                self.log(f"[卖出信号] 短均线 下穿 长均线 -> 准备做空, Close={close_price:.2f}")
                self.buy_bracket_with_atr(is_long=False)

        else:
            # ============ 已有持仓 -> 交给 bracket 止盈止损处理 ============
            pass

    def buy_bracket_with_atr(self, is_long=True):
        """
        用 bracket_order 下单，并根据 ATR 动态计算止盈止损距离。
        """
        close_price = self.dataclose[0]
        atr_value = self.atr[0]

        # 1) 计算止损、止盈价
        stop_dist = self.p.atr_stop_loss * atr_value
        tp_dist = self.p.atr_take_profit * atr_value

        if is_long:
            entry_price = close_price
            stop_price = entry_price - stop_dist
            limit_price = entry_price + tp_dist
        else:
            entry_price = close_price
            stop_price = entry_price + stop_dist
            limit_price = entry_price - tp_dist

        # 2) 资金管理：根据单笔风险来计算下单手数
        total_value = self.broker.getvalue()
        risk_amount = total_value * self.p.risk_per_trade

        if is_long:
            risk_per_share = entry_price - stop_price
        else:
            risk_per_share = stop_price - entry_price

        if risk_per_share <= 0:
            self.log("[警告] 风险距离 <= 0, 无法下单。检查 ATR 或价格逻辑。")
            return

        size = int(risk_amount / risk_per_share)
        if size <= 0:
            self.log("[警告] 计算出的下单手数 <= 0, 跳过。")
            return

        # 3) 提交 bracket_order
        if is_long:
            self.log(
                f"[提交买Bracket] Buy Price={entry_price:.2f}, Stop={stop_price:.2f}, TP={limit_price:.2f}, Size={size}"
            )
            self.order = self.buy_bracket(
                size=size,
                price=entry_price,
                stopprice=stop_price,
                limitprice=limit_price,
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
            )

    def stop(self):
        self.log(f"[回测结束] 最终市值: {self.broker.getvalue():.2f}")


class DMAStrategyIntradayImproved(bt.Strategy):
    """
    改良版 DMA 策略示例（仅做多 & 日内交易 & 距离阈值过滤）:
      1) 仅做多，不做空；
      2) 日内平仓，不隔夜 -> 到尾盘(示例为15:55)强制平仓；
      3) 当短期均线上穿长期均线且收盘价高于长期均线一定比例(距离阈值)时，才确认买入；
      4) 用 bracket_order + ATR 动态止盈止损 + 单笔风险管理。
    """

    params = (
        # --- 均线参数 ---
        ('fast_period', 10),
        ('slow_period', 20),
        # --- 距离阈值过滤，例如 0.01 表示需要价格高出长均线 1% 才算有效突破
        ('distance_threshold', 0.01),
        # --- ATR 止盈止损参数 ---
        ('atr_period', 14),
        ('atr_stop_loss', 1.5),
        ('atr_take_profit', 3.0),
        # --- 资金管理 ---
        ('risk_per_trade', 0.01),  # 单笔风险占总资金的 1%
        # --- 日内平仓时间设置(示例: 美股15:55) ---
        ('intraday_close_hour', 15),
        ('intraday_close_minute', 55),
    )

    def log(self, txt, dt=None):
        """自定义日志函数，调试/回测时可打印输出"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {txt}")

    def __init__(self):
        # 收盘价引用
        self.dataclose = self.datas[0].close

        # 短期均线 & 长期均线
        self.ma_fast = bt.indicators.SMA(self.dataclose,
                                         period=self.p.fast_period)
        self.ma_slow = bt.indicators.SMA(self.dataclose,
                                         period=self.p.slow_period)

        # 均线交叉：>0 表示短均线由下向上穿越长均线 (金叉)；<0 表示短均线下穿 (死叉)
        self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)

        # ATR 用于动态止盈止损
        self.atr = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)

        # 跟踪当前订单（避免重复下单）
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"[成交] 买单执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                )
            else:
                self.log(
                    f"[成交] 卖单执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                )
            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("[警告] 订单取消/保证金不足/拒绝")
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"[交易结束] 毛收益: {trade.pnl:.2f}, 净收益: {trade.pnlcomm:.2f}")

    def next(self):
        """策略主逻辑，每个新 Bar 调用一次。"""
        if self.order:
            # 如果有在途订单，则不重复生成新订单
            return

        # 1) 先检查是否临近收盘 -> 强制日内平仓
        dt = self.datas[0].datetime.datetime(0)
        if dt.hour == self.p.intraday_close_hour and dt.minute >= self.p.intraday_close_minute:
            # 若当前还持仓，则全部平仓
            if self.position:
                self.log(f"[日内平仓] {dt} 强制平仓")
                self.order = self.close()
            return  # 不再执行下方开仓逻辑

        # 2) 若数据还不够长（初始 warm-up 或者 ATR 需要的历史），直接跳过
        min_period_needed = max(self.p.fast_period, self.p.slow_period,
                                self.p.atr_period)
        if len(self) < min_period_needed:
            return

        # ============ 策略只做多，不做空 ============
        if not self.position:
            # 检查金叉：短均线上穿长均线
            if self.crossover[0] > 0:
                # 检查“距离阈值”过滤：当前收盘价须比长均线高出一定比例
                slow_ma_val = self.ma_slow[0]
                close_price = self.dataclose[0]

                # 例如要求 close_price > slow_ma_val * (1 + distance_threshold)
                if close_price > slow_ma_val * (1.0 +
                                                self.p.distance_threshold):
                    self.log(
                        f"[买入信号] DMA金叉 + 收盘价高于长均线{self.p.distance_threshold*100:.1f}% -> 准备开多，Close={close_price:.2f}"
                    )
                    self.buy_bracket_with_atr()
        else:
            # 已持有多仓 -> 交给 bracket 止盈止损，或日内强制平仓来管理
            pass

    def buy_bracket_with_atr(self):
        """
        使用 bracket_order 下多单，并根据 ATR 动态止盈止损 + 单笔风险。
        """
        close_price = self.dataclose[0]
        atr_value = self.atr[0]

        stop_dist = self.p.atr_stop_loss * atr_value  # 止损距离
        tp_dist = self.p.atr_take_profit * atr_value  # 止盈距离

        # 多单设置
        entry_price = close_price
        stop_price = entry_price - stop_dist
        limit_price = entry_price + tp_dist

        # 资金管理：单笔风险
        total_value = self.broker.getvalue()
        risk_amount = total_value * self.p.risk_per_trade
        risk_per_share = entry_price - stop_price

        if risk_per_share <= 0:
            self.log("[警告] 风险距离 <= 0, 无法下单。检查 ATR 或价格逻辑。")
            return

        size = int(risk_amount / risk_per_share)
        if size <= 0:
            self.log("[警告] 计算出的下单手数 <= 0, 跳过。")
            return

        self.log(
            f"[提交BuyBracket] Buy@{entry_price:.2f}, SL={stop_price:.2f}, TP={limit_price:.2f}, Size={size}"
        )
        self.order = self.buy_bracket(
            size=size,
            price=entry_price,  # 主多单
            stopprice=stop_price,  # 止损单
            limitprice=limit_price,  # 止盈单
        )

    def stop(self):
        """回测结束时输出最终市值"""
        self.log(f"[回测结束] 最终市值: {self.broker.getvalue():.2f}")


class DMABollPartialIntradayStrategy(bt.Strategy):
    """
    DMA + 布林带分批开仓 + 分批止盈 + ATR追踪止损（做多示例） + 日内平仓（基于美股收盘时间）。

    核心变化：不再通过“比较下一根Bar日期”来判断日内平仓，而是直接通过当前Bar的时间
             (如15:55之后就平仓)，避免实盘中无法预知下一根Bar导致的误判。
    """

    params = (
        # --- DMA 参数 ---
        ('fast_period', 15),
        ('slow_period', 50),
        # --- 布林带参数 (用于加仓触发) ---
        ('bb_period', 20),
        ('bb_devfactor', 2.0),
        # --- 分批开仓相关 ---
        ('position_size', 0.9),  # 总目标仓位占比（相对账户净值）
        ('batch_ratio1', 0.5),  # 第一批占比
        ('batch_ratio2', 0.5),  # 第二批占比
        # --- 止盈止损相关 ---
        ('use_atr', True),  # 是否使用 ATR 动态止盈/追踪止损
        ('atr_period', 14),
        ('atr_mult_stop', 2.0),  # 追踪止损 ATR 倍数
        ('tp1_mult', 1.0),  # 第一档止盈倍数
        ('tp2_mult', 2.0),  # 第二档止盈倍数
        # 如果不想用 ATR, 也可改用固定比例
        ('fixed_stop_loss', 0.02),
        ('fixed_tp1', 0.02),
        ('fixed_tp2', 0.04),
        # --- 日内平仓相关 ---
        # 美股常规交易时间：09:30 ~ 16:00，这里示例在15:55之后就平仓
        ('close_hour', 15),  # 尾盘平仓的小时
        ('close_minute', 55),  # 尾盘平仓的分钟
    )

    def log(self, txt, dt=None):
        """统一的日志输出函数，可自行关闭或定制。"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {txt}")

    def __init__(self):
        # 注意改用 self.dataclose，而不是 self.close，否则会覆盖 self.close() 方法
        self.dataclose = self.datas[0].close

        # 1) DMA
        self.ma_fast = bt.indicators.SMA(self.dataclose,
                                         period=self.p.fast_period)
        self.ma_slow = bt.indicators.SMA(self.dataclose,
                                         period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)

        # 2) 布林带，用于加仓阈值
        self.boll = bt.indicators.BollingerBands(self.dataclose,
                                                 period=self.p.bb_period,
                                                 devfactor=self.p.bb_devfactor)

        # 3) ATR，用于动态止盈/止损
        if self.p.use_atr:
            self.atr = bt.indicators.ATR(self.datas[0],
                                         period=self.p.atr_period)

        # 订单跟踪 & 阶段标记
        self.order = None
        self.buy_stage = 0  # 0=未开仓, 1=开了第一批, 2=开满第二批
        self.tp_stage = 0  # 0=未止盈, 1=完成第一档, 2=完成第二档
        self.entry_price = None
        self.total_size = 0
        self.trail_stop = None

    def next(self):
        # 如果有在途订单，则不下新单
        if self.order:
            return

        # ---------------------------
        # (A) 正常策略交易逻辑部分
        # ---------------------------
        price = self.dataclose[0]
        cross_val = self.crossover[0]

        # 若当前无持仓，则重置标记
        if self.position.size <= 0:
            self.buy_stage = 0
            self.tp_stage = 0
            self.total_size = 0
            self.trail_stop = None
            self.entry_price = None

        # A.1 未开仓 & DMA 金叉 -> 开第一批多仓
        if self.buy_stage == 0 and cross_val > 0:
            batch1_size = self.calc_batch_size(self.p.batch_ratio1)
            if batch1_size > 0:
                self.log(f"[开仓] DMA金叉 -> 买入第一批, size={batch1_size}")
                self.order = self.buy(size=batch1_size)

        # A.2 已开第一批 & 价格突破布林带上轨 -> 开第二批多仓
        elif self.buy_stage == 1:
            if price > self.boll.top[0]:
                batch2_size = self.calc_batch_size(self.p.batch_ratio2)
                if batch2_size > 0:
                    self.log(f"[加仓] 价格上破布林带上轨, buy size={batch2_size}")
                    self.order = self.buy(size=batch2_size)

        # A.3 持仓后，执行分批止盈 & 追踪止损
        if self.position.size > 0:
            self.manage_position()

        # ---------------------------
        # (B) 日内平仓：基于当前时间判断
        # ---------------------------
        current_dt = self.data.datetime.datetime(0)
        if (current_dt.hour > self.p.close_hour) or \
           (current_dt.hour == self.p.close_hour and current_dt.minute >= self.p.close_minute):
            # 已过了收盘前指定时间 -> 平仓
            if self.position.size > 0:
                self.log(f"[日内平仓] 当前时间 {current_dt}, 不留隔夜 -> 平仓")
                self.order = self.close()

    def manage_position(self):
        price = self.dataclose[0]

        # 1) 计算止损距离
        if self.p.use_atr:
            current_atr = self.atr[0]
            stop_loss_dist = self.p.atr_mult_stop * current_atr
        else:
            current_atr = None
            stop_loss_dist = self.p.fixed_stop_loss * self.entry_price

        # 2) 更新/设置追踪止损
        new_trail_stop = price - stop_loss_dist
        if not self.trail_stop:
            self.trail_stop = self.entry_price - stop_loss_dist
        else:
            self.trail_stop = max(self.trail_stop, new_trail_stop)

        # 如果价格 <= 追踪止损，则全部平仓
        if price <= self.trail_stop:
            self.log(
                f"[追踪止损触发] price={price:.2f} <= trail_stop={self.trail_stop:.2f}, 全部平仓"
            )
            self.order = self.close()
            return

        # 3) 分批止盈
        if self.tp_stage < 2:
            if self.p.use_atr:
                tp1_price = self.entry_price + self.p.tp1_mult * current_atr
                tp2_price = self.entry_price + self.p.tp2_mult * current_atr
            else:
                tp1_price = self.entry_price * (1.0 + self.p.fixed_tp1)
                tp2_price = self.entry_price * (1.0 + self.p.fixed_tp2)

            # 第1档止盈
            if self.tp_stage == 0 and price >= tp1_price:
                sell_size = math.floor(self.position.size * 0.5)
                self.log(
                    f"[止盈1] price={price:.2f} >= {tp1_price:.2f}, 卖出 {sell_size} 股做部分止盈"
                )
                self.order = self.sell(size=sell_size)

            # 第2档止盈
            elif self.tp_stage == 1 and price >= tp2_price:
                sell_size = math.floor(self.position.size * 0.3)
                self.log(
                    f"[止盈2] price={price:.2f} >= {tp2_price:.2f}, 卖出 {sell_size} 股, 剩余留给追踪止损"
                )
                self.order = self.sell(size=sell_size)

    def calc_batch_size(self, ratio):
        """
        根据 ratio 计算分批下单股数:
        ratio 为 self.position_size 的分割占比。
        例如 ratio=0.5 表示 (总目标持仓) * 0.5。
        """
        total_value = self.broker.getvalue()
        target_shares = (total_value *
                         self.p.position_size) / self.dataclose[0]
        batch_shares = target_shares * ratio
        return max(0, math.floor(batch_shares))

    def notify_order(self, order):
        # 订单状态变更
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"[成交] 买单执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                )
                self.total_size += order.executed.size

                # 开第一批 -> buy_stage=1
                if self.buy_stage == 0:
                    self.buy_stage = 1
                    self.entry_price = order.executed.price
                # 开第二批 -> buy_stage=2
                elif self.buy_stage == 1:
                    self.buy_stage = 2

            else:  # 卖单
                self.log(
                    f"[成交] 卖单执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                )
                self.total_size -= abs(order.executed.size)
                if self.total_size <= 0:
                    # 清仓完毕
                    self.log("[平仓完成] 持仓已清空")
                    self.buy_stage = 0
                    self.tp_stage = 0
                    self.entry_price = None
                    self.trail_stop = None
                else:
                    # 若是部分止盈
                    if self.tp_stage < 2:
                        # 第一次止盈 => tp_stage=1
                        if self.tp_stage == 0:
                            self.tp_stage = 1
                        # 第二次止盈 => tp_stage=2
                        elif self.tp_stage == 1:
                            self.tp_stage = 2

            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("[警告] 订单取消/保证金不足/拒绝")
            self.order = None

    def notify_trade(self, trade):
        # 当一笔交易完全结束时，会回调这里
        if trade.isclosed:
            self.log(f"[交易结束] 毛收益: {trade.pnl:.2f}, 净收益: {trade.pnlcomm:.2f}")

    def stop(self):
        # 回测结束时，输出最终净值
        final_val = self.broker.getvalue()
        self.log(f"[回测结束] 最终市值: {final_val:.2f}")
