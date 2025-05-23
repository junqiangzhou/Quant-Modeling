from rl.single_shot.trading_env import StockTradingEnv
from feature.feature import compute_online_feature
from config.config import Action, label_names, buy_sell_signals
from data.stocks_fetcher import MAG7
from strategy.rule_based import should_buy, should_sell, calc_pred_labels
from model.model import compute_model_output

from numpy.typing import NDArray
import numpy as np
import torch


class BacktestSingleShot(StockTradingEnv):

    def __init__(self,
                 stock: str,
                 start_date: str,
                 end_date: str,
                 init_fund: float = 1.0e4):
        super().__init__(stock, start_date, end_date, init_fund)
        print(f">>>>>{stock}:")
        self.debug_mode = False
        self.use_gt_label = False
        if self.use_gt_label:
            print("+++++++++++++Using GT labels for backtest++++++++++++++")

    def compute_action(self) -> Action:
        date = self.current_step
        # Must sell all shares before earnings day
        if date in self.stock_data.index and self.stock_data.loc[date][
                "Earnings_Date"]:
            if self.debug_mode:
                print(f"Earnings day must sell, {date}")
            return Action.Sell

        # Must sell to cut-loss
        if date in self.stock_data.index:
            cost_base = self.cost_base
            price = self.stock_data.loc[date]["Close"]
            if price < cost_base * 0.92:
                if self.debug_mode:
                    print(f"Must cut loss and sell, {date}")
                return Action.Sell

        # Call the model
        buy_sell_signals_vals = self.stock_data.loc[date,
                                                    buy_sell_signals].values
        price_above_ma = self.stock_data.loc[date, "Price_Above_MA_5"] == 1
        price_below_ma = self.stock_data.loc[date, "Price_Below_MA_5"] == 1

        if self.use_gt_label:
            pred = self.stock_data.loc[date, label_names].values
        else:
            features = compute_online_feature(self.stock_data, date)
            probs, pred = compute_model_output(self.prediction_model, features)
            if probs is None or pred is None:
                return Action.Hold

        if should_sell(pred, buy_sell_signals_vals,
                       price_below_ma):  # need to sell
            if self.debug_mode:
                print(
                    f"------Predicted to sell, {date}, close price {price:.2f}, prob. of trending down {probs[:, 2]}"
                )
            return Action.Sell
        elif should_buy(pred, buy_sell_signals_vals,
                        price_above_ma):  # good to buy
            if self.debug_mode:
                print(
                    f"++++++Predicted to buy, {date}, close price {price:.2f}, prob. of trending up {probs[:, 1]}"
                )
            return Action.Buy
        else:
            return Action.Hold

    def run(self) -> None:
        done = False
        print("current_date: ", self.start_date, " end_date: ", self.end_date)
        # print("current_date: ", self.start_date, " end_date: ", self.end_date)
        try:
            start_price, end_price = self.stock_data.loc[self.start_date][
                "Close"], self.stock_data.loc[self.end_date]["Close"]
            print(
                f"Price change: {(end_price - start_price) / start_price * 100: .2f} %",
                end="      ")
        except:
            print(f"Failed to compute price change {stock}")
        while not done:
            if self.debug_mode:
                self.render()
            action = self.compute_action()
            obs, reward, done, _, _ = self.step(int(action.value))

        print(f"Quant profit: {reward * 100: .2f} %")

        return


if __name__ == "__main__":
    start_date = "2021-01-01"
    end_date = "2021-12-31"
    init_fund = 1.0e4

    for stock in MAG7 + ["QQQ", "SPY"]:
        backtest = BacktestSingleShot(stock, start_date, end_date, init_fund)
        backtest.run()
