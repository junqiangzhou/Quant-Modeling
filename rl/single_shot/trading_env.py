from data.data_fetcher import create_dataset, get_date_back
from feature.feature import compute_online_feature
from model.model import PredictionModel
from config.config import (ENCODER_TYPE, feature_names, look_back_window,
                           label_feature)

import gym
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
import bisect
from gym import spaces


class StockTradingEnv(gym.Env):

    def __init__(self,
                 stock: str,
                 start_date: str,
                 end_date: str,
                 init_fund: float = 1.0e4):
        super(StockTradingEnv, self).__init__()

        shifted_start_date = get_date_back(start_date, look_back_window + 30)
        self.stock_data = create_dataset(stock, shifted_start_date, end_date)

        # Load prediction model
        self.prediction_model = PredictionModel(feature_len=len(feature_names),
                                                seq_len=look_back_window,
                                                encoder_type=ENCODER_TYPE)
        self.prediction_model.load_state_dict(torch.load('./model/model.pth'))
        self.prediction_model.eval()

        self.initial_balance = init_fund
        self.balance = init_fund
        self.stock_holdings = 0
        self.portfolio = init_fund

        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        start_index = bisect.bisect_left(self.stock_data.index, start_date)
        self.start_date = self.stock_data.index[start_index]
        self.end_date = self.stock_data.index[-1]
        self.current_step = self.start_date

        # Define action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # 1) stock price (1)
        # 2) predicted probability (3)
        # 3) holdings (1)
        # 4) balance (1)
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

    def reset(self):
        """Reset the environment at the start of an episode."""
        self.current_step = self.start_date
        self.balance = self.initial_balance
        self.stock_holdings = 0
        return self._next_observation()

    def _next_observation(self):
        """Get next observation (stock price, predicted probability, holdings, balance)."""
        row = self.stock_data.loc[self.current_step]

        features = compute_online_feature(self.stock_data, self.current_step)
        if features is None:
            raise ValueError("Error in computing features")

        features_tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            logits = self.prediction_model(features_tensor)
            logits = logits.reshape(len(label_feature), 3)
            predicted_prob = torch.softmax(
                logits,
                dim=1).float().numpy()  # convert logits to probabilities

        obs = np.concatenate(([row['Close']], predicted_prob[0, :],
                              [self.stock_holdings,
                               self.balance])).astype(np.float32)

        return obs

    def step(self, action):
        """Execute an action (buy/sell/hold) and compute the reward."""
        row = self.stock_data.loc[self.current_step]
        stock_price = row['Close']

        if action == 1 and self.balance >= stock_price:  # Buy
            num_buy = self.balance // stock_price
            self.stock_holdings += num_buy
            self.balance -= stock_price * num_buy
        elif action == 2 and self.stock_holdings > 0:  # Sell
            num_sell = self.stock_holdings
            self.stock_holdings -= num_sell
            self.balance += stock_price * num_sell

        done = self.current_step >= self.end_date
        self.current_step = self.next_step(self.current_step)

        # Reward: Portfolio value change
        self.portfolio = self.balance + (self.stock_holdings * stock_price)
        reward = (self.portfolio - self.initial_balance) / self.initial_balance

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        """Render the current state (for debugging)."""
        print(
            f'Step: {self.current_step}, Balance: {self.balance:.2f}, Holdings: {self.stock_holdings:.1f}, Portfolio: {self.portfolio:.2f}'
        )
