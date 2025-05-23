from data.data_fetcher import create_dataset
from data.utils import get_date_back, normalize_date
from feature.feature import compute_online_feature
from feature.label import compute_labels
from model.model import PredictionModel
from config.config import (ENCODER_TYPE, Action, feature_names,
                           look_back_window, label_names, MODEL_EXPORT_NAME)

import gymnasium as gym
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
import bisect
from gymnasium import spaces


class StockTradingEnv(gym.Env):

    def __init__(self,
                 stock: str,
                 start_date: str,
                 end_date: str,
                 init_fund: float = 1.0e4):
        super(StockTradingEnv, self).__init__()

        # print(
        #     f"Stocks: {stock}, start date: {start_date}, end date: {end_date}")
        # Load stock data
        shifted_start_date = get_date_back(start_date, look_back_window + 30)
        df = create_dataset(stock, shifted_start_date, end_date)
        df, _ = compute_labels(df)
        self.stock_data = df

        # Load prediction model
        self.prediction_model = PredictionModel(feature_len=len(feature_names),
                                                seq_len=look_back_window,
                                                encoder_type=ENCODER_TYPE)
        self.prediction_model.load_state_dict(
            torch.load(f"./model/export/{MODEL_EXPORT_NAME}.pth"))
        self.prediction_model.eval()

        self.init_balance = init_fund
        self.balance = init_fund
        self.stock_holdings = 0  # Number of shares holding
        self.cost_base = 0  # purchase price
        self.portfolio = init_fund

        start_date = normalize_date(start_date)
        while start_date not in self.stock_data.index:
            start_date += timedelta(days=1)
        self.start_date = start_date
        self.end_date = self.stock_data.index[-1]
        self.current_step = self.start_date
        self.price_scale = self.stock_data.loc[self.start_date][
            'Close']  # Normalize observation

        # Define action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # 1) stock price (1) -- Normalized
        # 2) predicted probability (3)
        # 3) holdings (1)
        # 4) balance (1) -- Normalized
        self.observation_space = spaces.Box(low=0,
                                            high=np.inf,
                                            shape=(6, ),
                                            dtype=np.float32)

    def next_step(self, current_step):
        next_step = current_step
        while next_step < self.end_date:
            next_step += timedelta(days=1)
            if next_step in self.stock_data.index:
                break
        return next_step

    def reset(self, seed=None, options=None):
        """Reset the environment at the start of an episode."""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.start_date
        self.balance = self.init_balance
        self.stock_holdings = 0
        self.cost_base = 0
        self.portfolio = self.init_balance
        return self._next_observation(), {}

    def _next_observation(self):
        """Get next observation (stock price, predicted probability, holdings, balance)."""
        row = self.stock_data.loc[self.current_step]

        features = compute_online_feature(self.stock_data, self.current_step)
        if features is None:
            raise ValueError("Error in computing features")

        features_tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            logits = self.prediction_model(features_tensor)
            logits = logits.reshape(len(label_names), 3)
            predicted_prob = torch.softmax(
                logits,
                dim=1).float().numpy()  # convert logits to probabilities

        obs = np.concatenate(
            ([row['Close'] / self.price_scale], predicted_prob[0, :],
             [self.stock_holdings,
              self.balance / self.init_balance])).astype(np.float32)

        return obs

    def step(self, action):
        """Execute an action (buy/sell/hold) and compute the reward."""
        row = self.stock_data.loc[self.current_step]
        stock_price = row['Close']

        if action == int(
                Action.Buy.value) and self.balance >= stock_price:  # Buy
            num_buy = self.balance // stock_price
            self.stock_holdings += num_buy
            self.balance -= stock_price * num_buy
            self.cost_base = stock_price
        elif action == int(
                Action.Sell.value) and self.stock_holdings > 0:  # Sell
            num_sell = self.stock_holdings
            self.stock_holdings -= num_sell
            self.balance += stock_price * num_sell
            self.cost_base = 0.0

        done = self.current_step >= self.end_date

        # Reward: Portfolio value change
        self.portfolio = self.balance + (self.stock_holdings * stock_price)
        reward = (self.portfolio - self.init_balance) / self.init_balance
        truncated = False

        self.current_step = self.next_step(self.current_step)
        return self._next_observation(), reward, done, truncated, {}

    def render(self, mode='human'):
        """Render the current state (for debugging)."""
        print(
            f'Step: {self.current_step}, Balance: {self.balance:.2f}, Holdings: {self.stock_holdings:.1f}, Portfolio: {self.portfolio:.2f},'
        )
