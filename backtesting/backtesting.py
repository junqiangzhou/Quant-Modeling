from data.data_fetcher import download_data
from data.stocks_fetcher import fetch_stocks
from feature.feature import look_back_window, feature_names
from feature.feature import compute_online_feature
from model.model import PredictionModel
from config.config import ENCODER_TYPE

from typing import List
from numpy.typing import NDArray
from datetime import datetime, timedelta
import torch
import pandas as pd
import numpy as np
from enum import Enum
import collections
import random

random_seed_test = 20


class Action(Enum):
    Hold = 0
    Buy = 1
    Sell = -1


class BacktestSystem:

    def __init__(self,
                 stock_lists: List[str],
                 start_date: str,
                 end_date: str,
                 init_fund: float = 1.0e4):
        self.stocks_data_pool = {}
        for stock in stock_lists:
            try:
                df = download_data(stock, start_date, end_date)
                df.index = df.index.date
                self.stocks_data_pool[stock] = df
            except ValueError:
                print(f"{stock} data not available")
        self.start_date = df.index[0]
        self.end_date = df.index[-1]

        # Load the saved parameters
        # Set to evaluation mode
        self.model = PredictionModel(feature_len=len(feature_names),
                                     seq_len=look_back_window,
                                     encoder_type=ENCODER_TYPE)
        self.model.load_state_dict(torch.load('./model/model.pth'))
        self.model.eval()

        self.init_fund = init_fund
        self.fund = init_fund
        self.action = Action.Hold
        self.stocks_hold = collections.defaultdict(int)
        self.cost_base = collections.defaultdict(float)

    def reset(self):
        self.fund = self.init_fund
        self.stocks_hold = collections.defaultdict(int)
        self.cost_base = collections.defaultdict(float)
        self.action = Action.Hold

    def compute_action(self, stock: str, date: datetime) -> Action:
        # Must sell all shares before earnings day
        if date in self.stocks_data_pool[stock].index and self.stocks_data_pool[
                stock].loc[date]["Earnings_Date"]:
            # print(f"Earnings day must sell, {date}")
            return Action.Sell

        # Must sell to cut-loss
        if date in self.stocks_data_pool[stock].index:
            cost_base = self.cost_base[stock]
            price = self.stocks_data_pool[stock].loc[date]["Close"]
            if price < cost_base * 0.92:
                return Action.Sell

        features = compute_online_feature(self.stocks_data_pool[stock], date)
        if features is not None:
            features_tensor = torch.tensor(features,
                                           dtype=torch.float32).unsqueeze(0)
        else:
            return Action.Hold

        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.sigmoid(
                logits).float().numpy()  # convert logits to probabilities

        probs_up = sum(probs[0, ::2]) / 3.0
        probs_down = sum(probs[0, 1:probs.shape[1] + 1:2]) / 3.0

        if probs_down > 0.5:  # need to sell
            # print(f"------Predicted to sell, {date}")
            return Action.Sell
        elif probs_up > 0.5:  # good to buy
            # print(f"++++++Predicted to buy, {date}")
            return Action.Buy
        else:
            return Action.Hold

    def buy(self, stock: str, date: datetime) -> None:
        if date < self.start_date:
            return

        price = self.stocks_data_pool[stock].loc[date]["Close"]
        if self.fund > 100:
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


if __name__ == "__main__":
    random.seed(random_seed_test)  # use different seed from data_fetcher
    stocks = fetch_stocks()
    stocks_testing = random.sample(stocks, 20)
    # stocks = ["TSLA", "AAPL", "GOOGL", "AMZN", "MSFT", "META", "NFLX", "NVDA"]
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    testing = BacktestSystem(stocks_testing, start_date, end_date)
    for stock in stocks_testing:
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
                end="   ")
        except:
            continue
        while current_date <= end_date:
            action = testing.compute_action(stock, current_date)
            if action == Action.Buy:
                testing.buy(stock, current_date)
            elif action == Action.Sell:
                testing.sell(stock, current_date)

            current_date += timedelta(days=1)
        print(
            f"Quant profit: {testing.get_profit(end_date) / testing.init_fund * 100: .2f} %"
        )
