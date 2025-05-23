from data.data_fetcher import create_dataset
from data.utils import get_date_back, normalize_date
from feature.feature import compute_online_feature
from feature.label import compute_labels
from model.model import PredictionModel
from strategy.rule_based import should_buy, should_sell, calc_pred_labels
from data.stocks_fetcher import MAG7
from config.config import (MODEL_EXPORT_NAME, ENCODER_TYPE, Action,
                           random_seed, label_names, buy_sell_signals,
                           look_back_window, feature_names)

from typing import List
from numpy.typing import NDArray
from datetime import datetime, timedelta, timezone
import torch
import pandas as pd
import numpy as np
from enum import Enum
import collections
import random
import bisect


# This file is outdated/duplicated and use backtest_single_shot.py instead
class BacktestSystem:

    def __init__(self,
                 stock_lists: List[str],
                 start_date: str,
                 end_date: str,
                 init_fund: float = 1.0e4):
        self.stocks_data_pool = {}
        for stock in stock_lists:
            try:
                shifted_start_date = get_date_back(start_date,
                                                   look_back_window + 30)
                df = create_dataset(stock, shifted_start_date, end_date)
                df, _ = compute_labels(df)
                self.stocks_data_pool[stock] = df
            except ValueError:
                print(f"{stock} data not available")
        start_date = normalize_date(start_date)
        while start_date not in df.index:
            start_date += timedelta(days=1)
        self.start_date = start_date
        self.end_date = df.index[-1]

        # Load the saved parameters
        # Set to evaluation mode
        self.model = PredictionModel(feature_len=len(feature_names),
                                     seq_len=look_back_window,
                                     encoder_type=ENCODER_TYPE)
        self.model.load_state_dict(
            torch.load(f"./model/export/{MODEL_EXPORT_NAME}.pth"))
        self.model.eval()

        self.init_fund = init_fund
        self.fund = init_fund
        self.action = Action.Hold
        self.stocks_hold = collections.defaultdict(int)
        self.cost_base = collections.defaultdict(float)
        self.use_gt_label = True
        if self.use_gt_label:
            print("+++++++++++++Using GT labels for backtest++++++++++++++")

    def reset(self):
        self.fund = self.init_fund
        self.stocks_hold = collections.defaultdict(int)
        self.cost_base = collections.defaultdict(float)
        self.action = Action.Hold

    def compute_action(self, stock: str, date: datetime) -> Action:
        df = self.stocks_data_pool[stock]
        if date not in df.index:
            # print(f"skip due to date {date} not in table")
            return Action.Hold

        # Must sell all shares before earnings day
        if df.loc[date]["Earnings_Date"]:
            if debug_mode:
                print(f"Earnings day must sell, {date}")
            return Action.Sell

        # Must sell to cut-loss
        cost_base = self.cost_base[stock]
        price = df.loc[date]["Close"]
        if price < cost_base * 0.92:
            if debug_mode:
                print(f"Must cut loss and sell, {date}")
            return Action.Sell

        buy_sell_signals_vals = df.loc[date, buy_sell_signals].values
        price_above_ma = df.loc[date, "Price_Above_MA_5"] == 1
        price_below_ma = df.loc[date, "Price_Below_MA_5"] == 1

        # Compute buy/sell labels
        if self.use_gt_label:
            pred = df.loc[date][label_names].values
        else:
            features = compute_online_feature(df, date)
            if features is None or np.isnan(features).any() or np.isinf(
                    features).any():
                # print(f"NaN or INF detected in {stock} on {date}")
                return Action.Hold
            else:
                features_tensor = torch.tensor(features, dtype=torch.float32)

            with torch.no_grad():
                logits = self.model(features_tensor)
                logits = logits.reshape(len(label_names), 3)
                probs = torch.softmax(
                    logits,
                    dim=1).float().numpy()  # convert logits to probabilities
                pred = calc_pred_labels(probs)

        if should_sell(pred, buy_sell_signals_vals,
                       price_below_ma):  # need to sell
            if debug_mode:
                print(
                    f"------Predicted to sell, {date}, close price {price:.2f}, prob. of trending down {probs[:, 2]}"
                )
            return Action.Sell
        elif should_buy(pred, buy_sell_signals_vals,
                        price_above_ma):  # good to buy
            if debug_mode:
                print(
                    f"++++++Predicted to buy, {date}, close price {price:.2f}, prob. of trending up {probs[:, 1]}"
                )
            return Action.Buy
        else:
            return Action.Hold

    def buy(self, stock: str, date: datetime) -> None:
        if date < self.start_date:
            return

        price = self.stocks_data_pool[stock].loc[date]["Close"]
        if self.fund > 100:
            if debug_mode:
                print(f">>>>>>>>>>>Execute to buy, {date}")
            shares = int(self.fund / price)
            self.stocks_hold[stock] += shares
            self.cost_base[stock] = price
            self.fund -= shares * price
        return

    def sell(self, stock: str, date: datetime) -> None:
        if date > self.end_date:
            return

        price = self.stocks_data_pool[stock].loc[date]["Close"]
        if self.stocks_hold[stock] > 0:
            if debug_mode:
                print(f"<<<<<<<<<<<<Execute to sell, {date}")
            shares = self.stocks_hold[stock]

            self.fund += shares * price
            self.stocks_hold[stock] = 0
            self.cost_base[stock] = 0

        return

    def get_profit(self, date: datetime) -> float:
        equity = 0
        for stock in self.stocks_hold:
            price = self.stocks_data_pool[stock].loc[date]["Close"]
            shares = self.stocks_hold[stock]
            equity += price * shares

        return self.fund + equity - self.init_fund

    def render(self, date):
        """Render the current state (for debugging)."""
        portfolio = self.get_profit(date) + self.init_fund
        shares = list(self.stocks_hold.values())
        cost_base = list(self.cost_base.values())
        print(
            f'Step: {date}, Balance: {self.fund:.2f}, Holdings: {shares[0] :.1f}, Portfolio: {portfolio:.2f}, Cost base: {cost_base[0]: .2f}'
        )


if __name__ == "__main__":
    random.seed(random_seed)  # use different seed from data_fetcher
    testing_stocks = MAG7

    debug_mode = False
    start_date = "2021-01-01"
    end_dates = ["2021-12-31"
                 ]  # ["2015-12-31", "2016-12-31", "2018-12-31", "2020-12-31"]

    # debug_mode = False
    # start_date = "2015-01-01"
    # end_dates = ["2015-12-31", "2016-12-31", "2018-12-31", "2020-12-31"]

    for end_date in end_dates:
        testing = BacktestSystem(testing_stocks, start_date, end_date)
        # print("current_date: ", start_date, " end_date: ", end_date)
        for stock in testing_stocks:
            testing.reset()

            if stock not in testing.stocks_data_pool:
                print(f"{stock} data not available")
                continue

            df = testing.stocks_data_pool[stock]
            current_date, end_date = testing.start_date, testing.end_date
            print(f">>>>>{stock}")
            try:
                start_price, end_price = df.loc[current_date]["Close"], df.loc[
                    end_date]["Close"]
                print(
                    f"Price change: {(end_price - start_price) / start_price * 100: .2f} %",
                    end="      ")
            except:
                print(f"Failed to compute price change {stock}")
                continue

            while current_date <= end_date:
                action = testing.compute_action(stock, current_date)
                if action == Action.Buy:
                    testing.buy(stock, current_date)
                elif action == Action.Sell:
                    testing.sell(stock, current_date)

                if debug_mode and current_date in testing.stocks_data_pool[
                        stock].index:
                    testing.render(current_date)

                current_date += timedelta(days=1)
            print(
                f"Quant profit: {testing.get_profit(end_date) / testing.init_fund * 100: .2f} %"
            )
