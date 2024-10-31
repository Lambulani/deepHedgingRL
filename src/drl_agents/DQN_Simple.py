import sys 
sys.path.append("/home/bndlev001/deepHedgingRL")
import gymnasium as gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import tensorflow as tf
from src.pricing.asset_price_process import GBM
from src.pricing.option_price_process import BSM
from src.custom_environments.HedgeEnv import env_hedging
from torch import nn

mu = 0
dt = 1/5
T = 10
num_steps = int(T/dt)
s_0 = float(100)
strike_price = s_0
sigma = 0.01
r = 0


apm = GBM(mu=mu, dt=dt, s_0=s_0, sigma=sigma)
opm = BSM(strike_price=strike_price, risk_free_interest_rate=r, volatility=sigma, T=T, dt=dt)

env = env_hedging(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.1,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL", shares_per_contract=100,
                  option_price_model=opm)




from stable_baselines3 import DQN
from gym import spaces
from typing import Callable, Tuple
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.distributions import DiagGaussianDistribution
from torch.nn import functional as F
import os 


n_envs = 10
vec_env = make_vec_env(lambda:env_hedging(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.1,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL", shares_per_contract=100,
                  option_price_model=opm), n_envs= n_envs)


# Set up directories for logging and saving models
log_dir = "/home/bndlev001/deepHedgingRL/results/DQN/du/logs"

models_dir = "/home/bndlev001/deepHedgingRL/models/DQN/du/DQN_2"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def linear_decay_schedule(initial_value: float, final_value:float) -> Callable[[float], float]:
    def schedule(progress: float) -> float:
        # Linearly decay the learning rate from `initial_value` to 0
        return initial_value - progress*(initial_value- final_value)
    return schedule

learning_rate_schedule = get_schedule_fn(linear_decay_schedule(1e-5, 1e-4))

 
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[128,128, 128, 128,128])

model = DQN(policy= "MlpPolicy", 
    env=vec_env,
    learning_rate=learning_rate_schedule,
    batch_size= 1024, 
    buffer_size= 750000,
    learning_starts= 10000,
    gamma = 0.87,
    target_update_interval= 100,
    train_freq= (15000, "episode"),
    gradient_steps= -1, 
    exploration_fraction= 0.16,
    exploration_final_eps= 0.02,
    policy_kwargs= policy_kwargs,
    verbose=1,
    device= "auto")

TIMESTEPS = 2000000
for i in range(1,1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN_2", progress_bar= True )
    model.save(f"{models_dir}/dqn_{TIMESTEPS*i}")

