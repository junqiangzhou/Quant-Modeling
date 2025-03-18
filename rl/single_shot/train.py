from rl.single_shot.trading_env import StockTradingEnv
from config.config import random_seed
from data.stocks_fetcher import MAG7

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import pandas as pd
import numpy as np

set_random_seed(random_seed)

# Create environment
start_dates = [
    "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"
]
end_dates = [
    "2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"
]
stock = "AAPL"


# Function to create environments
def make_env(stock, start_date, end_date):

    def _init():
        return StockTradingEnv(stock, start_date, end_date)

    return _init


env_fns = [
    make_env(stock, start_dates[i], end_dates[i])
    for i in range(len(start_dates)) for stock in MAG7
]

# Single process
envs = DummyVecEnv(env_fns)

# Train RL agent
model = PPO("MlpPolicy", envs, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save("./rl/single_shot/ppo_stock_trader")
