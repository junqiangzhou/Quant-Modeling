from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from rl.multi_shot.trading_env import StockTradingEnv
from config.config import random_seed
from data.stocks_fetcher import MAG7

set_random_seed(random_seed)

# Load trained model
model = PPO.load("./rl/multi_shot/ppo_stock_trader")

start_date = "2020-01-01"
end_date = "2020-12-31"
env = StockTradingEnv(MAG7, start_date, end_date)

# Test RL agent
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
