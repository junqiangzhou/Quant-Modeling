import backtrader as bt

from model.model import PredictionModel
from config.config import (MODEL_EXPORT_NAME, ENCODER_TYPE, feature_names,
                           look_back_window, Action, label_names,
                           buy_sell_signals)
from feature.feature import compute_online_feature, normalize_features

import torch
from datetime import datetime
import numpy as np
from numpy.typing import NDArray


class MLStrategy(bt.Strategy):
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
        ('target_pct', 0.9),
        ('stop_loss', 0.08),  # 2% 止损
        ('take_profit', 0.50),  # 5% 止盈
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

        # Load the saved parameters
        # Set to evaluation mode
        self.model = PredictionModel(feature_len=len(feature_names),
                                     seq_len=look_back_window,
                                     encoder_type=ENCODER_TYPE)
        self.model.load_state_dict(
            torch.load(f"./model/export/{MODEL_EXPORT_NAME}.pth"))
        self.model.eval()

        self.debug_mode = False

    def notify_order(self, order):
        pass

    def notify_trade(self, trade):
        pass

    def next(self):
        pass

    def stop(self):
        """回测结束时输出最终市值"""
        self.log(f"最终市值: {self.broker.getvalue():.2f}")

    def compute_features(self):
        history = np.array([
            getattr(self.data, feature_names)[-i]
            for i in reversed(range(look_back_window))
        ])
        if np.isnan(history).any() or np.isinf(history).any():
            return None
        features = np.expand_dims(history, axis=0)
        features_scaled = normalize_features(features)
        return features_scaled

    def compute_action(self) -> Action:
        # Ensure enough historical data is available
        if len(self.data) < look_back_window:
            return Action.Hold

        # Avoid multiple open orders
        if self.order:
            return Action.Hold

        # Must sell all shares before earnings day
        if getattr(self.data, "Earnings_Date")[0]:
            if self.debug_mode:
                print(
                    f"Earnings day must sell, {self.data.datetime.datetime(0)}"
                )
            return Action.Sell

        features = self.compute_features()
        if features is None or np.isnan(features).any() or np.isinf(
                features).any():
            if self.debug_mode:
                print(
                    f"NaN or INF detected on {self.data.datetime.datetime(0)}")
                return Action.Hold
        else:
            features_tensor = torch.tensor(features, dtype=torch.float32)

        buy_sell_signals_vals = [
            getattr(self.data, col)[0] for col in buy_sell_signals
        ]
        bullish_signal = getattr(self.data, "Price_Above_MA_5")[0]
        bearish_signal = getattr(self.data, "Price_Below_MA_5")[0]
        with torch.no_grad():
            logits = self.model(features_tensor)
            logits = logits.reshape(len(label_names), 3)
            probs = torch.softmax(
                logits,
                dim=1).float().numpy()  # convert logits to probabilities

        def should_buy(probs: NDArray) -> bool:
            pred = np.argmax(probs, axis=1)
            trend_up_labels = np.sum(pred == 1)
            trend_up_indicators = np.sum(buy_sell_signals_vals == 1)
            if trend_up_labels == len(
                    label_names
            ) and trend_up_indicators >= 1 and bullish_signal == 1:
                return True

            return False

        def should_sell(probs: NDArray) -> bool:
            pred = np.argmax(probs, axis=1)
            trend_down_labels = np.sum(pred == 2)
            trend_down_indicators = np.sum(buy_sell_signals_vals == -1)
            if trend_down_labels == len(
                    label_names
            ) and trend_down_indicators >= 1 and bearish_signal == 1:
                return True

            return False

        if should_sell(probs):  # need to sell
            if self.debug_mode:
                print(
                    f"------Predicted to sell, {self.data.datetime.datetime(0)}, close price {self.Close[0]:.2f}, prob. of trending down {probs[:, 2]}"
                )
            return Action.Sell
        elif should_buy(probs):  # good to buy
            if self.debug_mode:
                print(
                    f"++++++Predicted to buy, {self.data.datetime.datetime(0)}, close price {self.Close[0]:.2f}, prob. of trending up {probs[:, 1]}"
                )
            return Action.Buy
        else:
            return Action.Hold
