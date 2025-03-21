from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from rl.single_shot.trading_env import StockTradingEnv
from config.config import random_seed
from data.stocks_fetcher import MAG7

set_random_seed(random_seed)

# Load trained model
model = PPO.load("./rl/single_shot/ppo_stock_trader", device='cpu')

start_date = "2018-01-01"
end_date = "2018-12-31"
for stock in MAG7:
    env = StockTradingEnv(stock, start_date, end_date)

    # Test RL agent
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        # env.render()

    start_price, end_price = env.stock_data.loc[
        env.start_date]["Close"], env.stock_data.loc[env.end_date]["Close"]
    print(
        f"Price change: {(end_price - start_price) / start_price * 100: .2f} %",
        end="    ")

    reward = (env.portfolio - env.init_balance) / env.init_balance
    print(f"Portfolio change: {reward * 100:.2f} %")
