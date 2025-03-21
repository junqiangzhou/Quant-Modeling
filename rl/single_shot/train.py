from rl.single_shot.trading_env import StockTradingEnv
from config.config import random_seed
from data.stocks_fetcher import MAG7

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import torch

device = 'cpu'
set_random_seed(random_seed)

# Create environment
start_dates = [
    "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"
]
end_dates = [
    "2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"
]


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
envs.seed(random_seed)
# envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
# envs.action_space.seed(random_seed)
# envs.observation_space.seed(random_seed)

# Train RL agent
policy_kwargs = dict(
    net_arch=[64, 64],  # Increase layers and neurons for complex problems
    activation_fn=torch.nn.ReLU)
# lr_schedule = lambda progress: 1e-3 * progress  # Linearly decrease LR

model = PPO(
    "MlpPolicy",
    envs,
    policy_kwargs=policy_kwargs,
    learning_rate=0.0003,
    ent_coef=0.01,
    # n_steps=4096,
    # n_epochs=50,
    batch_size=128,
    gamma=0.95,
    verbose=1,
    clip_range=0.2,
    device=device)
model.learn(total_timesteps=500000)

# Save model
model.save("./rl/single_shot/ppo_stock_trader")
