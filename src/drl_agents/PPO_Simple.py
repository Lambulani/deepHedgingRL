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

env = env_hedging(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.01,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL", shares_per_contract=100,
                  option_price_model=opm)




from stable_baselines3 import PPO
from gym import spaces
from typing import Callable, Tuple
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.distributions import DiagGaussianDistribution
from torch.nn import functional as F
import os 


n_envs = 10
vec_env = make_vec_env(lambda:env_hedging_ppo(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.01,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL", shares_per_contract=100,
                  option_price_model=opm), n_envs= n_envs)


# Set up directories for logging and saving models
log_dir = "/home/bndlev001/deepHedgingRL/results/PPO/du/logs"

models_dir = "/home/bndlev001/deepHedgingRL/models/PPO/du/PPO_14"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def linear_decay_schedule(initial_value: float, final_value:float) -> Callable[[float], float]:
    def schedule(progress: float) -> float:
        # Linearly decay the learning rate from `initial_value` to 0
        return initial_value - progress*(initial_value- final_value)
    return schedule

learning_rate_schedule = get_schedule_fn(linear_decay_schedule(1e-5, 1e-4))

 
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[64,128, 128, 128,64], vf=[64, 128, 128, 128, 64]))
# Instantiate and train the PPO model as per the Du paper
model = PPO(
    policy= "MlpPolicy", 
    policy_kwargs= policy_kwargs, 
    env= vec_env, 
    learning_rate=learning_rate_schedule,
    n_steps= 75000, #update every 750,000 episodes
    n_epochs = 5,
batch_size = 1000, 
    clip_range= 0.24286367561098068, 
clip_range_vf = 0.155591002226219580,
    verbose=1,
    gae_lambda= 0.9871191601388827, 
    gamma= 0.8790853310508997,
    ent_coef=0.2,  # c2
    vf_coef=0.5,  #c1
   normalize_advantage= True,
    tensorboard_log=log_dir,
    device= "auto"
)


# Train the model in increments and save after each block of timesteps
TIMESTEPS = 750000
for i in range(1,1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_14", progress_bar= True )
    model.save(f"{models_dir}/ppo_{TIMESTEPS*i}")



