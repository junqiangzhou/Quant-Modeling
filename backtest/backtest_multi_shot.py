from rl.multi_shot.trading_env import StockTradingEnv
from feature.feature import compute_online_feature
from config.config import Action, label_names, buy_sell_signals
from data.stocks_fetcher import MAG7

from numpy.typing import NDArray
from typing import List, Tuple
import numpy as np
import torch


class BacktestSingleShot(StockTradingEnv):

    def __init__(self,
                 stocks: List[str],
                 start_date: str,
                 end_date: str,
                 init_fund: float = 1.0e4):
        super().__init__(stocks, start_date, end_date, init_fund)
        self.debug_mode = False

    def compute_action(self) -> Tuple[int, Action]:
        date = self.current_step

        def should_sell(probs: NDArray, buy_sell_signals_vals,
                        bearish_signal) -> bool:
            pred = np.argmax(probs, axis=1)
            trend_down_labels = np.sum(pred == 2)
            trend_down_indicators = np.sum(buy_sell_signals_vals == -1)
            if trend_down_labels == len(
                    label_names
            ) and trend_down_indicators >= 1 and bearish_signal == 1:
                return True

            return False

        def should_buy(probs: NDArray, buy_sell_signals_vals,
                       bullish_signal) -> bool:
            pred = np.argmax(probs, axis=1)
            trend_up_labels = np.sum(pred == 1)
            trend_up_indicators = np.sum(buy_sell_signals_vals == 1)
            if trend_up_labels == len(
                    label_names
            ) and trend_up_indicators >= 1 and bullish_signal == 1:
                return True

            return False

        # Check if needed to sell current holdings
        if self.stock_holdings > 0:
            stock = self.stocks[self.stock_held]
            # Must sell all shares before earnings day
            if date in self.stock_data[stock].index and self.stock_data[
                    stock].loc[date]["Earnings_Date"]:
                if self.debug_mode:
                    print(f"Earnings day must sell, {date}")
                return (self.stock_held, Action.Sell)

            # Must sell to cut-loss
            if date in self.stock_data[stock].index:
                cost_base = self.cost_base
                price = self.stock_data[stock].loc[date]["Close"]
                if price < cost_base * 0.92:
                    if self.debug_mode:
                        print(f"Must cut loss and sell, {date}")
                    return (self.stock_held, Action.Sell)

            # Check sell probability/signal
            features = compute_online_feature(self.stock_data[stock], date)
            if features is None or np.isnan(features).any() or np.isinf(
                    features).any():
                # print(f"NaN or INF detected in {stock} on {date}")
                return (self.stock_held, Action.Hold)
            else:
                features_tensor = torch.tensor(features, dtype=torch.float32)

            buy_sell_signals_vals = self.stock_data[stock].loc[
                date, buy_sell_signals].values
            bearish_signal = self.stock_data[stock].loc[date,
                                                        "Price_Below_MA_5"]

            with torch.no_grad():
                logits = self.prediction_model(features_tensor)
                logits = logits.reshape(len(label_names), 3)
                probs = torch.softmax(
                    logits,
                    dim=1).float().numpy()  # convert logits to probabilities

            if should_sell(probs, buy_sell_signals_vals,
                           bearish_signal):  # need to sell
                if self.debug_mode:
                    print(
                        f"------Predicted to sell, {date}, close price {price:.2f}, prob. of trending down {probs[:, 2]}"
                    )
                return (self.stock_held, Action.Sell)

        else:  # Currently holds no stocks, and check if needed to buy
            # Find the stock with highest probability of trending up
            max_probs = None
            max_5day_buy_prob = 0
            max_stock = None
            for stock in self.stocks:
                features = compute_online_feature(self.stock_data[stock], date)
                if features is None or np.isnan(features).any() or np.isinf(
                        features).any():
                    # print(f"NaN or INF detected in {stock} on {date}")
                    return (self.stock_held, Action.Hold)
                else:
                    features_tensor = torch.tensor(features,
                                                   dtype=torch.float32)

                with torch.no_grad():
                    logits = self.prediction_model(features_tensor)
                    logits = logits.reshape(len(label_names), 3)
                    probs = torch.softmax(logits, dim=1).float().numpy(
                    )  # convert logits to probabilities
                    if probs[0, 1] > max_5day_buy_prob:
                        max_5day_buy_prob = probs[0, 1]
                        max_probs = probs
                        max_stock = stock

            buy_sell_signals_vals = self.stock_data[max_stock].loc[
                date, buy_sell_signals].values
            bullish_signal = self.stock_data[max_stock].loc[date,
                                                            "Price_Above_MA_5"]

            if should_buy(max_probs, buy_sell_signals_vals,
                          bullish_signal):  # good to buy
                if self.debug_mode:
                    print(
                        f"++++++Predicted to buy, {date}, close price {price:.2f}, prob. of trending up {probs[:, 1]}"
                    )
                return (self.stocks.index(max_stock), Action.Buy)

        return (self.stock_held, Action.Hold)

    def run(self) -> None:
        done = False
        print("current_date: ", self.start_date, " end_date: ", self.end_date)
        while not done:
            if self.debug_mode:
                self.render()
            stock_index, order = self.compute_action()
            action = (stock_index, int(order.value))
            obs, reward, done, _ = self.step(action)

        print(f"Quant profit: {reward * 100: .2f} %")

        return


if __name__ == "__main__":
    stocks = MAG7
    start_date = "2021-01-01"
    end_date = "2021-12-31"
    init_fund = 1.0e4

    backtest = BacktestSingleShot(stocks, start_date, end_date, init_fund)
    backtest.run()
