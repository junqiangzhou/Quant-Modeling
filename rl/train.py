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
model.learn(total_timesteps=10000)

# Save model
# model.save("ppo_stock_trader")

# Load trained model
# model = PPO.load("ppo_stock_trader")

# Test RL agent
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
