#Import Dependencies 
import gym
import sys 
sys.path.append("/home/bndlev001/deepHedgingRL")
import numpy as np
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
from src.pricing.asset_price_process import GBM
from src.pricing.option_price_process import BSM
from src.custom_environments.HedgeEnv import env_hedging
from torch import nn
import time


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
env = env_hedging(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 1, tick_size=0.01,
                     L=1, strike_price=strike_price, int_holdings=True, initial_holding=0, mode="PL",moneyness = "atm",
                  option_price_model=opm)


from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.utils import get_schedule_fn
import os

#Custom Feature Extractor with LSTM
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 201):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Assuming the input is a 1D vector, reshape it to (batch_size, sequence_length, features)
        input_dim = observation_space.shape[0]

        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.fc_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape the input to add a sequence length of 1
        observations = observations.unsqueeze(1)
        lstm_out, _ = self.lstm(observations)
        # Use the output of the last time step
        last_timestep_output = lstm_out[:, -1, :]
        return self.fc_net(last_timestep_output)

# Custom DQN Policy
class CustomDQNPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Discrete,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=None,
        normalize_images=False,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
    ):
        super(CustomDQNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

# Learning rate schedule
    
lr_schedule = get_schedule_fn(1e-4)

from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super(RewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self) -> bool:
        # Save the reward for this step
        reward = self.locals["rewards"][0]  # Access the current reward
        if self.n_calls % self.check_freq == 0:
            self.rewards.append(reward)  # Log rewards at each check frequency
        return True

# Usage
callback = RewardCallback(check_freq=1000, verbose=1)

models_dir = "/home/bndlev001/deepHedgingRL/models/DQN"
logdir = "/home/bndlev001/deepHedgingRL/results/DQN/logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

model_DQN = DQN(policy= CustomDQNPolicy, 
    env=env,
    learning_rate=lr_schedule,
    verbose=1,
    gamma= 0.85, 
    batch_size=32,
    tensorboard_log= logdir)

TIMESTEPS = 3000000
model_DQN.learn(total_timesteps= TIMESTEPS, reset_num_timesteps= False, tb_log_name="DQN", callback= callback)
model_DQN.save(f"{models_dir}/{TIMESTEPS}")
