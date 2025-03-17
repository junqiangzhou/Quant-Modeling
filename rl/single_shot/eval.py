from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from rl.single_shot.trading_env import StockTradingEnv
from config.config import random_seed

set_random_seed(random_seed)

# Load trained model
model = PPO.load("./rl/single_shot/ppo_stock_trader")

start_date = "2022-01-01"
end_date = "2022-12-31"
env = StockTradingEnv("TSLA", start_date, end_date)

# Test RL agent
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
