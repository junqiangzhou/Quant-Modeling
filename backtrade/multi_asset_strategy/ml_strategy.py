import backtrader as bt

from model.model import PredictionModel, compute_model_output
from config.config import (MODEL_EXPORT_NAME, ENCODER_TYPE, feature_names,
                           look_back_window, Action, ActionProbability,
                           label_names, buy_sell_signals)
from feature.feature import compute_online_feature, normalize_features
from strategy.rule_based import should_buy, should_sell, calc_pred_labels

import torch
from datetime import datetime
import numpy as np
from numpy.typing import NDArray


class MultiAssetMLStrategy(bt.Strategy):
    """
    ML 策略示例：
    - 使用机器学习模型进行预测，决定买入或卖出。
    - 开仓后自动设置固定止盈和固定止损单（可选）。
    
    参数：
    - model: 机器学习模型
    - target_pct: 每次开仓的目标资金占比
    - stop_loss: 固定止损百分比（对开仓价）
    - take_profit: 固定止盈百分比（对开仓价）
    """

    params = (
        ('model', None),  # 机器学习模型
        ('debug_mode', False),  # Enable debug mode and logging
        ('use_gt_label', False),  # Use ground truth labels for backtest
        ('target_pct', 0.5),
        ('daily_change_perc', 0.05),
        ('earning_day_sell', False),  # Force to sell before earning day
        ('take_profit', False),
    )

    def log(self, txt, dt=None):
        """自定义日志函数，可在 debug 或回测时使用"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {txt}")

    def get_signals(self):
        """获取所有交易信号"""
        return {
            'buy': self.buy_signals,
            'sell': self.sell_signals,
            'position_size': self.position_sizes
        }

    def log_order_type(self, order):
        stock = order.data._name
        if order.exectype == bt.Order.Market:
            self.log(f"Market order {stock} executed at {order.executed.price}")
        elif order.exectype == bt.Order.Limit:
            self.log(f"Limit order {stock} executed at {order.executed.price}")
        elif order.exectype == bt.Order.Stop:
            self.log(f"Stop order {stock} executed at {order.executed.price}")
        elif order.exectype == bt.Order.StopLimit:
            self.log(f"Stop-Limit order {stock} executed at {order.executed.price}")
        return

    def __init__(self):
        # ========== 1. 保存引用 ==========
        self.start_date = None

        self.stop_loss = self.p.daily_change_perc * 2.0
        self.take_profit = self.p.daily_change_perc * 12.0 if self.p.take_profit else float(
            'inf')
        # ========== 2. 跟踪订单和止盈止损单 ==========
        self.orders = {}  # 主订单
        # self.stop_loss_orders = None  # 止损单
        # self.take_profit_orders = None  # 止盈单
        self.buy_signals = []  # 买入信号列表，格式为 (datetime, price)
        self.sell_signals = []  # 卖出信号列表，格式为 (datetime, price)
        self.position_sizes = []  # 持仓变化列表，格式为 (datetime, size)

        self.entry_prices = {}
        # Load the saved parameters
        # Set to evaluation mode
        self.model = PredictionModel(feature_len=len(feature_names),
                                     seq_len=look_back_window,
                                     encoder_type=ENCODER_TYPE)
        self.model.load_state_dict(
            torch.load(f"./model/export/{MODEL_EXPORT_NAME}.pth"))
        self.model.eval()

        self.debug_mode = self.p.debug_mode
        self.use_gt_label = self.p.use_gt_label
        self.earning_day_sell = self.p.earning_day_sell
        if self.use_gt_label:
            self.log("+++++++++++++Using GT labels for backtest++++++++++++++")

    def notify_order(self, order):
        """订单状态更新回调"""
        if order.status in [order.Submitted, order.Accepted]:
            # 订单提交/接受后，不做特殊处理
            return
        
        stock = order.data._name
        # 订单完成
        if order.status in [order.Completed]:
            # if self.debug_mode:
            #     self.log_order_type(order)
            if order.isbuy():
                if self.debug_mode:
                    self.log(
                        f"[成交] 买单执行: stock={stock}, 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                    )

                # 设置止盈止损单
                buy_price = order.executed.price
                size = order.executed.size
                sl_price = buy_price * (1 - self.stop_loss)
                tp_price = buy_price * (1 + self.take_profit)

                self.entry_prices[stock] = buy_price
                # self.stop_loss_order = self.sell(size=size,
                #                                  exectype=bt.Order.Stop,
                #                                  price=sl_price)
                # self.take_profit_order = self.sell(size=size,
                #                                    exectype=bt.Order.Limit,
                #                                    price=tp_price)
                # Collect buy signal
                self.buy_signals.append(
                    (self.datas[0].datetime.datetime(0), order.executed.price))
                self.position_sizes.append(
                    (self.datas[0].datetime.datetime(0), self.position.size))

            elif order.issell():
                if self.debug_mode:
                    self.log(
                        f"[成交] 卖单执行: stock={stock}, 价格={order.executed.price:.2f}, 数量={order.executed.size}"
                    )

                del self.entry_prices[stock]
                # 取消止盈止损单（如果尚未成交）
                # if self.stop_loss_order and self.stop_loss_order != order:
                #     self.cancel(self.stop_loss_order)
                # if self.take_profit_order and self.take_profit_order != order:
                #     self.cancel(self.take_profit_order)

                self.sell_signals.append(
                    (self.datas[0].datetime.datetime(0), order.executed.price))
                self.position_sizes.append(
                    (self.datas[0].datetime.datetime(0), self.position.size))

            # self.order = None

        # 订单取消/保证金不足/拒绝
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.debug_mode:
                if order in list(self.orders.values()):
                    self.log(
                        f"[警告] cancel order: {order.status}, {order.info}")
                # if order == self.stop_loss_order:
                #     self.log(
                #         f"[警告] cancel stop loss order: {order.status}, {order.info}"
                #     )
                # if order == self.take_profit_order:
                #     self.log(
                #         f"[警告] cancel take profit order: {order.status}, {order.info}"
                #     )
            # self.order = None
            # self.stop_loss_order = None
            # self.take_profit_order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        if self.debug_mode:
            self.log(
                f"TRADE PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}"
            )

    def next(self):
        # Skip if we have an active main order (buy/sell)
        # if self.order:
        # if self.debug_mode:
        #     self.log(f"Skipping,  we have an active order")
        # return

        current_date = self.datas[0].datetime.datetime(0)
        if self.start_date is None:
            self.start_date = current_date

        if self.start_date is None or (
                current_date - self.start_date).days < look_back_window:
            # if self.debug_mode:
            #     self.log(
            #         f"Warm-up: skipping, only {len(self.data)} bars available")
            return

        # if self.position.size == 0:
        #     if self.stop_loss_order:
        #         self.cancel(self.stop_loss_order)
        #     if self.take_profit_order:
        #         self.cancel(self.take_profit_order)

        stock_actions = {}
        for data in self.datas:
            stock = data._name
            position = self.getposition(data)

            # get model prediction
            action_prob = self.compute_action_prob(data)
            stock_actions[stock] = action_prob

            # determine buy/sell/hold strategy
            if position.size > 0:
                if action_prob.action == Action.Sell:
                    self.orders[stock] = self.sell(size=position.size,
                                                   price=data.close[0],
                                                   exectype=bt.Order.Close)
            else:
                if action_prob.action == Action.Buy:
                    cash = self.broker.get_cash()
                    cash = max(0, cash)
                    size = int(self.p.target_pct * cash / data.close[0])
                    size = max(0, size)
                    if size > 0:
                        self.orders[stock] = self.buy(size=size,
                                                      price=data.close[0],
                                                      exectype=bt.Order.Market)

    def stop(self):
        """回测结束时输出最终市值"""
        if self.debug_mode:
            self.log(f"最终市值: {self.broker.getvalue():.2f}")

    def compute_features(self, data) -> NDArray:
        history = np.array(
            [[getattr(data, name)[-i] for name in feature_names]
             for i in reversed(range(look_back_window))])
        if np.isnan(history).any() or np.isinf(history).any():
            return None
        features = np.expand_dims(history, axis=0)
        features_scaled = normalize_features(features)
        return features_scaled

    def compute_action_prob(self, data) -> ActionProbability:
        buy_sell_signals_vals = np.array(
            [getattr(data, col)[0] for col in buy_sell_signals])
        price_above_ma = getattr(data, "Price_Above_MA_5")[0] == 1
        price_below_ma = getattr(data, "Price_Below_MA_5")[0] == 1

        # Compute buy/sell labels
        if self.use_gt_label:
            pred = np.array([getattr(data, col)[0] for col in label_names])
            probs_avg = np.ones(3)
        else:
            features = self.compute_features(data)
            probs, pred, _ = compute_model_output(self.model, features)
            if probs is None or pred is None:
                if self.debug_mode:
                    self.log(
                        f"NaN or INF detected on {data.datetime.datetime(0)}")
                return ActionProbability(action=Action.Hold, prob=0)
            probs_avg = np.mean(probs, axis=0)

        if should_sell(pred, buy_sell_signals_vals,
                       price_below_ma):  # need to sell
            return ActionProbability(action=Action.Sell, prob=probs_avg[2])
        elif should_buy(pred, buy_sell_signals_vals,
                        price_above_ma):  # good to buy
            return ActionProbability(action=Action.Buy, prob=probs_avg[1])
        else:
            return ActionProbability(action=Action.Hold, prob=probs_avg[0])
