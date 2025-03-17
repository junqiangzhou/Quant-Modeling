from rl.trading_env import StockTradingEnv

from stable_baselines3 import PPO
import pandas as pd
import numpy as np

# Create environment
start_date = "2020-01-01"
end_date = "2020-12-31"
env = StockTradingEnv("AAPL", start_date, end_date)

# Train RL agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save("./rl/ppo_stock_trader")
