#Import the dependencies 
import gym
import numpy as np
import sys 
sys.path.append("C:/Users/levyb/Documents/Masters Data Science - 2nd Year/deepHedgingRL")
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
from src.pricing.asset_price_process import GBM
from src.pricing.option_price_process import BSM
from src.custom_environments.HedgeEnv_PPO import env_hedging_ppo
from torch import nn
import time


#Load hedging environment 
mu = 0
dt = 1/5
T = 10
num_steps = int(T/dt)
s_0 = float(100)
strike_price = s_0
sigma = 0.01
r = 0

def cost(delta_h, multiplier):
    TickSize = 0.1
    return multiplier * TickSize * (np.abs(delta_h) + 0.01 * delta_h**2)

apm = GBM(mu=mu, dt=dt, s_0=s_0, sigma=sigma)
opm = BSM(strike_price=strike_price, risk_free_interest_rate=r, volatility=sigma, T=T, dt=dt)
env = env_hedging_ppo(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.01,
                     L=1, strike_price=strike_price, int_holdings=True, initial_holding=0, mode="PL",
                  option_price_model=opm)

#Train Soft Actor Critic 

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_schedule_fn
import torch.nn as nn
import torch
import gym
import os 

# Custom Feature Extractor with 5 hidden layers and batch normalization
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 201):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]  # Assuming the input is a 1D vector

        self.fc_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),  
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  
            nn.ReLU(),
            
            nn.Linear(128, features_dim),
            nn.BatchNorm1d(features_dim), 
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.fc_net(observations)
    
# Learning rate schedule
lr_schedule = get_schedule_fn(1e-4)

# Custom callback to log rewards during training
from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super(RewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]  # Access the current reward
        if self.n_calls % self.check_freq == 0:
            self.rewards.append(reward)  # Log rewards at each check frequency
        return True

# Set up callback and directories for logging and saving models
callback = RewardCallback(check_freq=1000, verbose=1)

models_dir = "models/SAC"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Instantiate and train the SAC model

model = SAC(
    policy="MlpPolicy", 
    env=env,  
    learning_rate=lr_schedule,
    verbose=1,
    gamma=0.99,  # Discount factor
    batch_size=64,  # Batch size for SAC
    buffer_size=100000,  # Replay buffer size
    train_freq=1,  # Frequency of training updates
    gradient_steps=1,  # Gradient steps after each training update
    policy_kwargs=dict(features_extractor_class=CustomFeatureExtractor),
    tensorboard_log=logdir
)

# Train the model in increments and save after each block of timesteps
TIMESTEPS = 100000
for i in range(1, 16):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC", callback=callback)
    model.save(f"{models_dir}/sac_{TIMESTEPS * i}")

