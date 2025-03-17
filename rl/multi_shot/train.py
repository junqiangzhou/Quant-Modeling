from rl.multi_shot.trading_env import StockTradingEnv
from config.config import random_seed, device
from data.stocks_fetcher import MAG7

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO

import pandas as pd
import numpy as np

set_random_seed(random_seed)

# Create environment
start_date = "2020-01-01"
end_date = "2020-12-31"
env = StockTradingEnv(MAG7, start_date, end_date)

# Train RL agent
model = PPO("MultiInputPolicy", env, verbose=1, device=device)
model.learn(total_timesteps=100000)

# Save model
model.save("./rl/multi_shot/ppo_stock_trader")
