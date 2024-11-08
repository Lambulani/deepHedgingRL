import sys 
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

env = env_hedging(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 1, tick_size=0.1,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL", kappa = 0.1, act_space_type="discrete", shares_per_contract=100,
                  option_price_model=opm)




from stable_baselines3 import DQN
from gym import spaces
from typing import Callable, Tuple
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy
from torch.nn import functional as F
import os 



n_envs = 10
vec_env = make_vec_env(lambda:env_hedging(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 5  , tick_size=0.1,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL", shares_per_contract=100,
                  option_price_model=opm), n_envs= n_envs)


# Set up directories for logging and saving models
log_dir = "results/DQN/du/logs"

models_dir = "models/DQN/du/DQN_4"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def linear_decay_schedule(initial_value: float, final_value:float) -> Callable[[float], float]:
    def schedule(progress: float) -> float:
        # Linearly decay the learning rate from `initial_value` to 0
        return initial_value - progress*(initial_value- final_value)
    return schedule

learning_rate_schedule = get_schedule_fn(linear_decay_schedule(1e-5, 1e-4))


class DuMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DuMLP, self).__init__()
        
        # Define a sequence of layers: Linear -> BatchNorm -> ReLU
        layers = []
        layer_size = 128
        for _ in range(5):  # 5 hidden layers
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
            input_dim = layer_size  # Update input dim for the next layer
        
        # Output layer to match the action space dimension
        layers.append(nn.Linear(layer_size, output_dim))
        
        # Wrap layers in a Sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)
    
    def set_training_mode(self, mode: bool):
        """Sets training mode to control batch normalization layers."""
        if mode:
            self.train()
        else:
            self.eval()


class DuDQNPolicy(DQNPolicy):
    def __init__(self, observation_space , action_space,
                  lr_schedule, **kwargs):
        super(DuDQNPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        # Create the custom feature extractor (MLP with batch norm and ReLU)
        input_dim = observation_space.shape[0]
        output_dim = action_space.n
        self.q_net = DuMLP(input_dim, output_dim)
        self.q_net_target = DuMLP(input_dim, output_dim)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.eval()

        # Set up optimizer for the Q-network
        self.optimizer = th.optim.Adam(self.q_net.parameters(), lr=1e-4)
    
    def forward(self, x: th.Tensor, deterministic : bool = True) -> th.Tensor:
        return self.q_net(x)
    
    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
            q_values = self.q_net(obs)
            actions = q_values.argmax(dim=1).unsqueeze(-1)
            return actions

    def update_target_network(self):
        self.q_net_target.load_state_dict(self.q_net.state_dict())
    
    def set_training_mode(self, mode: bool):
        self.q_net.set_training_mode(mode)
        self.q_net_target.set_training_mode(mode)


model = DQN(policy= DuDQNPolicy, 
    env=vec_env,
    learning_rate=learning_rate_schedule,
    batch_size= 1024, 
    buffer_size= 500000,
    learning_starts= 50000,
    gamma = 0.9,
    target_update_interval= 750000,
    train_freq= ( 750000, "step"),
    gradient_steps= 1, 
    exploration_fraction= 0.2,
    exploration_initial_eps= 1, 
    exploration_final_eps= 0.01,
    verbose=1,
    device= "auto", 
    tensorboard_log= log_dir)

TIMESTEPS = 100000
for i in range(1, 500):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN_4", progress_bar= True)
    model.save(f"{models_dir}/dqn_{TIMESTEPS*i}")
