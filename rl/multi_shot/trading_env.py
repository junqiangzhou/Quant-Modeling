from data.data_fetcher import create_dataset
from data.utils import get_date_back, normalize_date
from feature.feature import compute_online_feature
from feature.label import compute_labels
from model.model import PredictionModel, compute_model_output
from config.config import (MODEL_EXPORT_NAME, ENCODER_TYPE, Action,
                           feature_names, look_back_window, label_names)

import gymnasium as gym
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from gymnasium import spaces


class StockTradingEnv(gym.Env):

    def __init__(self,
                 stocks: List[str],
                 start_date: str,
                 end_date: str,
                 init_fund: float = 1.0e4):
        super(StockTradingEnv, self).__init__()

        # Load stock data
        print(
            f"Stocks: {stocks}, start date: {start_date}, end date: {end_date}"
        )
        self.stock_data = {}
        shifted_start_date = get_date_back(start_date, look_back_window + 30)
        for stock in stocks:
            try:
                df = create_dataset(stock, shifted_start_date, end_date)
                df, _ = compute_labels(df)
                self.stock_data[stock] = df
            except:
                print(f" Error in processing {stock}")
                continue
        self.stocks = list(self.stock_data.keys())
        self.num_stocks = len(self.stocks)

        # Load prediction model
        self.prediction_model = PredictionModel(feature_len=len(feature_names),
                                                seq_len=look_back_window,
                                                encoder_type=ENCODER_TYPE)
        self.prediction_model.load_state_dict(
            torch.load(f"./model/export/{MODEL_EXPORT_NAME}.pth"))
        self.prediction_model.eval()

        self.init_balance = init_fund
        self.balance = init_fund
        self.stock_holdings = 0  # Number of shares held
        self.stock_held = 0  # Index of the stock held
        self.cost_base = 0.0
        self.portfolio = init_fund

        self.index = df.index
        start_date = normalize_date(start_date)
        while start_date not in self.index:
            start_date += timedelta(days=1)
        self.start_date = start_date
        self.end_date = self.index[-1]
        self.current_step = self.start_date

        # Define action space - Only 1 action per step:
        # action, stock (n) with action 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.MultiDiscrete([self.num_stocks, 3])

        # Observation space:
        # Continuous:
        # 1) stock price (n)
        # 2) predicted probability (n*3)
        # 3) holdings: # of shares (1)
        # 4) balance (1)
        # Discrete:
        # 1) stock (n)
        continuous_obv_space = spaces.Box(low=0,
                                          high=np.inf,
                                          shape=(4 * self.num_stocks + 2, ),
                                          dtype=np.float32)
        discrete_obv_space = spaces.Discrete(self.num_stocks)
        self.observation_space = spaces.Dict({
            "continuous": continuous_obv_space,
            "discrete": discrete_obv_space
        })

    def next_step(self, current_step):
        next_step = current_step
        while next_step < self.end_date:
            next_step += timedelta(days=1)
            if next_step in self.index:
                break
        return next_step

    def reset(self, seed=None, options=None):
        """Reset the environment at the start of an episode."""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.start_date
        self.balance = self.init_balance
        self.stock_holdings = 0
        self.stock_held = 0
        self.cost_base = 0
        self.portfolio = self.init_balance
        return self._next_observation(), {}

    def _next_observation(self):
        """Get next observation (stock price, predicted probability, holdings, balance)."""
        prices = np.zeros(self.num_stocks)
        predicted_probs = np.zeros((self.num_stocks, 3))
        for i in range(self.num_stocks):
            stock = self.stocks[i]
            row = self.stock_data[stock].loc[self.current_step]
            features = compute_online_feature(self.stock_data[stock],
                                              self.current_step)
            probs, pred, _ = compute_model_output(self.prediction_model,
                                                  features)
            if probs is None or pred is None:
                raise ValueError("Error in computing features")

            prices[i] = row['Close']
            predicted_probs[i] = probs[0, :]

        holdings = np.array([self.stock_holdings, self.balance])
        obs = {
            "continuous":
            np.concatenate((prices, predicted_probs.flatten(),
                            holdings)).astype(np.float32),
            "discrete":
            self.stock_held
        }

        return obs

    def step(self, action):
        """Execute an action (buy/sell/hold) and compute the reward."""
        stock_index, order = action[0], action[1]
        stock = self.stocks[stock_index]
        row = self.stock_data[stock].loc[self.current_step]
        stock_price = row['Close']

        def sell_stock(stock_index: int, stock_price: float):
            # can't sell given stock if not holding
            if self.stock_holdings == 0 or stock_index != self.stock_held:
                return

            num_sell = self.stock_holdings
            self.stock_holdings -= num_sell
            self.balance += stock_price * num_sell
            self.stock_held = 0
            self.cost_base = 0.0

        def buy_stock(stock_index: int, stock_price: float):
            # can't buy before selling the holdings
            if self.stock_holdings > 0 or self.balance < stock_price:
                return

            num_buy = self.balance // stock_price
            self.stock_holdings += num_buy
            self.balance -= stock_price * num_buy
            self.stock_held = stock_index
            self.cost_base = stock_price

        if order == int(Action.Buy.value):  # Buy
            buy_stock(stock_index, stock_price)
        elif order == int(
                Action.Sell.value) and self.stock_holdings > 0:  # Sell
            sell_stock(stock_index, stock_price)

        done = self.current_step >= self.end_date
        self.current_step = self.next_step(self.current_step)

        # Reward: Portfolio value change
        self.portfolio = self.balance
        if self.stock_holdings > 0:
            stock_held = self.stocks[self.stock_held]
            stock_held_price = self.stock_data[stock_held].loc[
                self.current_step]['Close']
            self.portfolio += self.stock_holdings * stock_held_price
        reward = (self.portfolio - self.init_balance) / self.init_balance
        truncated = False

        return self._next_observation(), reward, done, truncated, {}

    def render(self, mode='human'):
        """Render the current state (for debugging)."""
        print(
            f'Step: {self.current_step}, Holdings: {self.stock_holdings:.1f}, Stock: {self.stocks[self.stock_held]} Portfolio: {self.portfolio:.2f}'
        )
