from data.data_fetcher import download_data, get_date_back
from data.label import one_hot_encoder
from data.stocks_fetcher import fetch_stocks
from feature.feature import look_back_window, feature_names
from feature.feature import compute_online_feature
from model.model import PredictionModel
from config.config import ENCODER_TYPE, random_seed

from typing import List
from numpy.typing import NDArray
from datetime import datetime, timedelta
import torch
import pandas as pd
import numpy as np
from enum import Enum
import collections
import random
import bisect


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
                shifted_start_date = get_date_back(start_date,
                                                   look_back_window + 30)
                df = download_data(stock, shifted_start_date, end_date)
                # Trim data within the interested time window
                df = df.loc[start_date:end_date]
                df.index = df.index.date
                df = one_hot_encoder(df)
                self.stocks_data_pool[stock] = df
            except ValueError:
                print(f"{stock} data not available")
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.start_date = df.index[bisect.bisect_left(df.index, start_date)]
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
            if debug_mode:
                print(f"Earnings day must sell, {date}")
            return Action.Sell

        # Must sell to cut-loss
        if date in self.stocks_data_pool[stock].index:
            cost_base = self.cost_base[stock]
            price = self.stocks_data_pool[stock].loc[date]["Close"]
            if price < cost_base * 0.92:
                if debug_mode:
                    print(f"Must cut loss and sell, {date}")
                return Action.Sell

        features = compute_online_feature(self.stocks_data_pool[stock], date)
        if features is None or np.isnan(features).any() or np.isinf(
                features).any():
            # print(f"NaN or INF detected in {stock} on {date}")
            return Action.Hold
        else:
            features_tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.sigmoid(
                logits).float().numpy()  # convert logits to probabilities

        def should_buy(probs: NDArray) -> bool:
            trend_up_probs = probs[0, ::2]
            # trend_down_probs = probs[0, 1:probs.shape[1] + 1:2]

            if min(trend_up_probs) >= 0.6:  # and trend_up_probs[-1] >= 0.5:
                return True

            return False

        def should_sell(probs: NDArray) -> bool:
            # trend_up_probs = probs[0, ::2]
            trend_down_probs = probs[0, 1:probs.shape[1] + 1:2]

            if min(trend_down_probs
                   ) >= 0.6:  # and trend_down_probs[-1] >= 0.5:
                return True

            return False

        if should_sell(probs):  # need to sell
            if debug_mode:
                print(
                    f"------Predicted to sell, {date}, close price {price:.2f}, prob. of trending down {probs[0, 1:probs.shape[1] + 1:2]}"
                )
            return Action.Sell
        elif should_buy(probs):  # good to buy
            if debug_mode:
                print(
                    f"++++++Predicted to buy, {date}, close price {price:.2f}, prob. of trending up {probs[0, ::2]}"
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


if __name__ == "__main__":
    random.seed(random_seed)  # use different seed from data_fetcher
    _, testing_stocks = fetch_stocks()
    # testing_stocks = random.sample(testing_stocks, 30)
    testing_stocks = [
        "TSLA"  #"TSLA", "AAPL", "GOOGL", "AMZN", "MSFT", "META", "NFLX", "NVDA"
    ]
    debug_mode = True
    start_date = "2020-01-01"
    end_date = "2020-12-31"

    testing = BacktestSystem(testing_stocks, start_date, end_date)
    print("current_date: ", start_date, " end_date: ", end_date)
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

            current_date += timedelta(days=1)
        print(
            f"Quant profit: {testing.get_profit(end_date) / testing.init_fund * 100: .2f} %"
        )
