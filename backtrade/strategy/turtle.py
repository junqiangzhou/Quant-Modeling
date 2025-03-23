import backtrader as bt


class TurtleStrategyImproved(bt.Strategy):
    """
    改良版海龟策略示例：
    1) 同时实现两套系统：System1(20/10) & System2(55/20)，可单独启用或都启用；
    2) 引入 2N 止损 + 分批加仓逻辑；
    3) 可选“突破失败过滤”，如上次突破亏损则跳过下一次同周期突破；
    4) 使用 Bracket Orders 或者手动 OCO 止损单进行“硬止损”。

    主要参数：
    - use_system1, use_system2 : 是否启用系统1/系统2
      (系统1默认 entry=20, exit=10；系统2默认 entry=55, exit=20)
    - atr_period : 计算 ATR 的周期 (默认14)
    - risk_pct : 每次开仓的初始风险占总资金比例 (默认2%)
    - stop_n : 2N 止损倍数，默认 2
    - max_units : 分批加仓的最大次数，默认 4
    - unit_scale : 加仓的级距倍数，默认 1 (每 1N 上涨/下跌后加一仓)
    - fail_break_filter : 是否启用“上次突破亏损->跳过下次突破”过滤
    """

    params = (
        # --- 两套系统的开关 ---
        ('use_system1', True),
        ('use_system2', True),
        # --- 默认参数可根据原海龟策略设置 ---
        ('entry_period_s1', 20),  # System1 入场周期
        ('exit_period_s1', 10),  # System1 出场周期
        ('entry_period_s2', 55),  # System2 入场周期
        ('exit_period_s2', 20),  # System2 出场周期
        ('atr_period', 14),
        ('risk_pct', 0.02),  # 单笔风险 2%
        ('stop_n', 2.0),  # 2N 止损
        ('max_units', 4),  # 最多分批加仓 4 次
        ('unit_scale', 1.0),  # 每 1N 波动加/减仓
        ('fail_break_filter', True),  # 是否启用失败突破过滤
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {txt}")

    def __init__(self):
        # 收盘价引用
        self.dataclose = self.datas[0].close

        # ATR 指标用于计算波动率 N
        self.atr = bt.ind.ATR(self.datas[0], period=self.p.atr_period)

        # ========== 定义两套系统的 Highest/Lowest ==========
        # 1) System1
        if self.p.use_system1:
            self.highest_entry_s1 = bt.ind.Highest(
                self.datas[0].high, period=self.p.entry_period_s1)
            self.lowest_entry_s1 = bt.ind.Lowest(self.datas[0].low,
                                                 period=self.p.entry_period_s1)
            self.highest_exit_s1 = bt.ind.Highest(self.datas[0].high,
                                                  period=self.p.exit_period_s1)
            self.lowest_exit_s1 = bt.ind.Lowest(self.datas[0].low,
                                                period=self.p.exit_period_s1)
        else:
            self.highest_entry_s1 = self.lowest_entry_s1 = None
            self.highest_exit_s1 = self.lowest_exit_s1 = None

        # 2) System2
        if self.p.use_system2:
            self.highest_entry_s2 = bt.ind.Highest(
                self.datas[0].high, period=self.p.entry_period_s2)
            self.lowest_entry_s2 = bt.ind.Lowest(self.datas[0].low,
                                                 period=self.p.entry_period_s2)
            self.highest_exit_s2 = bt.ind.Highest(self.datas[0].high,
                                                  period=self.p.exit_period_s2)
            self.lowest_exit_s2 = bt.ind.Lowest(self.datas[0].low,
                                                period=self.p.exit_period_s2)
        else:
            self.highest_entry_s2 = self.lowest_entry_s2 = None
            self.highest_exit_s2 = self.lowest_exit_s2 = None

        # ========== 记录每个系统的持仓状态/加仓次数等 ==========
        # 真实海龟可以把每个系统看成独立的子仓位，这里简单用字典或对象记录
        self.sys_state = {
            's1': {
                'pos': 0,  # 当前持仓量（>0 多头, <0 空头, 0 无仓）
                'avg_price': 0,  # 持仓均价
                'units': 0,  # 已加仓次数
                'last_break_won': True  # 上次突破是否盈利 (用于失败突破过滤)
            },
            's2': {
                'pos': 0,
                'avg_price': 0,
                'units': 0,
                'last_break_won': True
            }
        }

        # 用于跟踪当前挂单
        # 如果用 bracket_order，会返回一组子订单；我们在 notify_order 里区别处理
        self.order_refs = {}  # key: order_ref -> some identification

    def notify_order(self, order):
        """处理订单状态更新"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            # 判断是哪一笔订单，更新系统状态
            if order.isbuy():
                self.log(
                    f"[成交] 买单: 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                )
            else:
                self.log(
                    f"[成交] 卖单: 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("[警告] 订单取消/保证金不足/拒绝")

        # 订单结束后，从记录里清除
        if not order.alive():
            if order.ref in self.order_refs:
                del self.order_refs[order.ref]

    def notify_trade(self, trade):
        """交易结束时输出盈亏，并标记该系统的上次突破是否盈利"""
        if trade.isclosed:
            self.log(f"[交易结束] System={trade.info.get('sysid','unknown')} "
                     f"毛收益: {trade.pnl:.2f}, 净收益: {trade.pnlcomm:.2f}")
            # 如果我们在下单时把 system id 放到 trade.info 里，可以在这里知道是哪个系统
            sysid = trade.info.get('sysid', None)
            if sysid and sysid in self.sys_state:
                # 若出现亏损，则标记 last_break_won=False（若启用 fail_break_filter）
                if trade.pnl < 0:
                    self.sys_state[sysid]['last_break_won'] = False
                else:
                    self.sys_state[sysid]['last_break_won'] = True

    def next(self):
        """
        每根K线结束后调用的核心逻辑
        1) 先检查历史数据长度是否足够
        2) 依次检查System1、System2的开平仓信号
        3) 更新子仓位后，再把多头/空头合并为一条实际指令(可简化处理)。
        """
        # 1) 数据长度检查
        min_bars_needed = max(
            (self.p.entry_period_s1 if self.p.use_system1 else 0),
            (self.p.entry_period_s2 if self.p.use_system2 else 0),
            self.p.atr_period)
        if len(self) < min_bars_needed:
            return

        # 2) 获取当前价格、ATR (N)
        current_price = self.dataclose[0]
        N = self.atr[0]  # 当日的 ATR
        if N <= 0:
            return

        # 分别处理 System1 / System2
        if self.p.use_system1:
            self.handle_system('s1', current_price, N,
                               self.highest_entry_s1[0],
                               self.lowest_entry_s1[0],
                               self.highest_exit_s1[0], self.lowest_exit_s1[0],
                               self.p.entry_period_s1, self.p.exit_period_s1)
        if self.p.use_system2:
            self.handle_system('s2', current_price, N,
                               self.highest_entry_s2[0],
                               self.lowest_entry_s2[0],
                               self.highest_exit_s2[0], self.lowest_exit_s2[0],
                               self.p.entry_period_s2, self.p.exit_period_s2)

        # 如果你想要把 System1 / System2 的仓位叠加再执行实际下单，可以在这里做“汇总”。
        # 但在 Backtrader 里更常用的做法是——直接让每个系统发自己的 buy_bracket() / sell_bracket()，
        # 并在 trade.info 里标记 sysid，这样就能独立跟踪盈亏和止损。

    def handle_system(self, sysid, price, N, highest_entry, lowest_entry,
                      highest_exit, lowest_exit, entry_period, exit_period):
        """
        针对某个系统的开平仓信号判断 + 分批加仓逻辑
        """
        state = self.sys_state[sysid]

        # ============== 若无持仓，寻找开仓时机 ==============
        if state['pos'] == 0:
            state['units'] = 0

            # 如果启用了 "fail_break_filter" 并且上次突破是亏损，则跳过这次信号
            if self.p.fail_break_filter and (not state['last_break_won']):
                return

            # 1) 多头突破：当前价 > (entry_period 日最高)
            if price > highest_entry:
                self.log(
                    f"[{sysid}] 多头开仓信号：收盘={price:.2f} 突破 {entry_period}日高")
                self.open_position(sysid, is_long=True, entry_price=price, N=N)

            # 2) 空头突破：当前价 < (entry_period 日最低)
            elif price < lowest_entry:
                self.log(
                    f"[{sysid}] 空头开仓信号：收盘={price:.2f} 跌破 {entry_period}日低")
                self.open_position(sysid,
                                   is_long=False,
                                   entry_price=price,
                                   N=N)

        # ============== 若有持仓，执行加仓/平仓 ==============
        else:
            pos_sign = 1 if state['pos'] > 0 else -1  # 多头:1, 空头:-1
            avg_price = state['avg_price']
            units = state['units']

            # --- 1) 分批加仓：仅在已有浮盈时加仓，且不超 max_units ---
            # 原海龟常设加仓触发点：每 0.5N 或 1N 上浮(多头)
            if units < self.p.max_units:
                # 多头 & 当前价较 avg_price 高出 unit_scale*N * (units) 才加仓
                if pos_sign > 0 and price > (avg_price +
                                             self.p.unit_scale * N * (units)):
                    self.log(f"[{sysid}] 多头加仓：当前价={price:.2f}, 第{units+1}次加仓")
                    self.add_position(sysid, is_long=True, N=N)
                # 空头 & 当前价较 avg_price 低出 unit_scale*N * (units) 才加仓
                elif pos_sign < 0 and price < (avg_price -
                                               self.p.unit_scale * N *
                                               (units)):
                    self.log(f"[{sysid}] 空头加仓：当前价={price:.2f}, 第{units+1}次加仓")
                    self.add_position(sysid, is_long=False, N=N)

            # --- 2) 退出信号：根据 exit_period 的反向突破平仓 ---
            # 多头：价格跌破 exit_period 日最低 -> 平仓
            if pos_sign > 0 and price < lowest_exit:
                self.log(
                    f"[{sysid}] 多头平仓信号：收盘={price:.2f} 跌破 {exit_period}日低 -> 全部平仓"
                )
                self.close_position(sysid)

            # 空头：价格突破 exit_period 日最高 -> 平仓
            elif pos_sign < 0 and price > highest_exit:
                self.log(
                    f"[{sysid}] 空头平仓信号：收盘={price:.2f} 突破 {exit_period}日高 -> 全部平仓"
                )
                self.close_position(sysid)

    def open_position(self, sysid, is_long, entry_price, N):
        """
        首次开仓 + 设置2N硬止损(Bracket Order) + 记录状态
        """
        state = self.sys_state[sysid]
        # 计算首仓size
        size = self.calc_unit_size(N)
        if size <= 0:
            self.log(f"[{sysid}] 计算出的size <= 0, 无法开仓")
            return

        stop_price = entry_price - self.p.stop_n * N if is_long else entry_price + self.p.stop_n * N
        limitprice = None  # 如果你想用追踪止盈可自定义

        self.log(
            f"[{sysid}] 首次{'多头' if is_long else '空头'}开仓：entry={entry_price:.2f}, stop={stop_price:.2f}, size={size}"
        )

        # 使用 buy_bracket / sell_bracket
        if is_long:
            o = self.buy_bracket(
                size=size,
                price=
                entry_price,  # 若想用市价单，可以 price=None + exectype=bt.Order.Market
                stopprice=stop_price,
                limitprice=limitprice,
                tradeinfo={'sysid': sysid}  # 让 trade.info 里带有 sysid
            )
            # o[0]是主单, o[1]是止损单, o[2]是止盈单(若有)
        else:
            o = self.sell_bracket(size=size,
                                  price=entry_price,
                                  stopprice=stop_price,
                                  limitprice=limitprice,
                                  tradeinfo={'sysid': sysid})

        # 更新子仓位状态
        pos_sign = 1 if is_long else -1
        state['pos'] = pos_sign * size
        state['avg_price'] = entry_price
        state['units'] = 1
        state[
            'last_break_won'] = True  # 先假设为 True，若最终这个trade亏损，会在 notify_trade() 里标记

    def add_position(self, sysid, is_long, N):
        """
        分批加仓：直接再发一个 bracket_order 或者单独发市价单+止损单
        这里示例用一个“合并止损”的做法就比较复杂了(需更新止损到新均价-2N)。
        为了简化，先只演示每次加仓都单独挂一个 2N 止损单。
        """
        state = self.sys_state[sysid]

        current_price = self.dataclose[0]
        new_size = self.calc_unit_size(N)
        if new_size <= 0:
            return

        # 新的止损价，理论上应该是(新仓平均价 - 2N)或(新仓平均价 + 2N)
        # 但原版海龟更常见的做法是：整体止损跟随第一笔的entry price±2N。
        # 此处示例：为新加仓单也下一个2N止损(多头则stop = current_price - 2N)。
        stop_price = current_price - self.p.stop_n * N if is_long else current_price + self.p.stop_n * N

        self.log(
            f"[{sysid}] 分批加仓：当前价={current_price:.2f}, stop={stop_price:.2f}, size={new_size}"
        )
        if is_long:
            o = self.buy_bracket(size=new_size,
                                 price=current_price,
                                 stopprice=stop_price,
                                 tradeinfo={'sysid': sysid})
            state['pos'] += new_size
        else:
            o = self.sell_bracket(size=new_size,
                                  price=current_price,
                                  stopprice=stop_price,
                                  tradeinfo={'sysid': sysid})
            state['pos'] -= new_size

        # 更新新的平均价(加权)，更新units
        total_shares = abs(state['pos'])
        old_shares = total_shares - new_size
        old_price = state['avg_price']
        new_avg = (old_price * old_shares +
                   current_price * new_size) / total_shares
        state['avg_price'] = new_avg
        state['units'] += 1

    def close_position(self, sysid):
        """
        平仓：直接 self.close()，让 Backtrader 自行撮合“反手”单
        也可以逐个取消此前的止损单，再发对冲单。
        """
        pos_size = self.sys_state[sysid]['pos']
        if pos_size == 0:
            return

        self.log(f"[{sysid}] 平仓：size={pos_size}")
        if pos_size > 0:
            # 多头 -> 直接 close() or sell(size=pos_size)
            self.close()  # 或 self.sell(size=pos_size)
        else:
            # 空头 -> 直接 close()
            self.close()  # 或 self.buy(size=abs(pos_size))

        # 记得重置系统状态
        self.sys_state[sysid]['pos'] = 0
        self.sys_state[sysid]['units'] = 0
        self.sys_state[sysid]['avg_price'] = 0

    def calc_unit_size(self, N):
        """
        计算单位头寸大小：risk_pct * total_value / (stop_n * N)
        因为 2N 是我们的总体止损宽度；若一次建仓被打止损，即亏 risk_pct * total_value
        """
        total_value = self.broker.getvalue()
        risk_amount = total_value * self.p.risk_pct
        stop_width = self.p.stop_n * N  # 2N
        if stop_width <= 0:
            return 0
        # 在期货实盘里要考虑合约乘数、最小交易单位，这里简单演示
        size = int(risk_amount / stop_width)
        return max(size, 0)

    def stop(self):
        self.log(f"[回测结束] 最终市值: {self.broker.getvalue():.2f}")
