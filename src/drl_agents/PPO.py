import sys 
sys.path.append("/home/bndlev001/deepHedgingRL")
import gymnasium as gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import tensorflow as tf
from src.pricing.asset_price_process import GBM
from src.pricing.option_price_process import BSM
from src.custom_environments.HedgeEnv_PPO import env_hedging_ppo
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

env = env_hedging_ppo(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.01,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL", shares_per_contract=100,
                  option_price_model=opm)




from stable_baselines3 import PPO
from gym import spaces
from typing import Callable, Tuple
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.env_util import make_vec_env
from torch.nn import functional as F
import os 


n_envs = 5
vec_env = make_vec_env(lambda:env_hedging_ppo(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.01,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL", shares_per_contract=100,
                  option_price_model=opm), n_envs= n_envs)



class Du_Custom_Network(nn.Module):
    
    def __init__(self, feature_dim: int ,last_layer_dim_pi: int = 201,last_layer_dim_vf: int = 1, hidden_dim: int = 165):
        super().__init__()

        # Save output dimensions, used to create the distributions
        self.hidden_dim = hidden_dim
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network with 5 hidden layers, batch normalization before ReLU
        self.policy_net = nn.Sequential(
            
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, last_layer_dim_pi)
        )

        # Value network with 5 hidden layers, batch normalization before ReLU
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, last_layer_dim_vf)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        logits = self.policy_net(features)
        return F.softmax(logits, dim=-1) 

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class Du_ActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor =Du_Custom_Network(self.features_dim)


# Set up directories for logging and saving models
log_dir = "/home/bndlev001/deepHedgingRL/results/PPO/du/logs"

models_dir = "/home/bndlev001/deepHedgingRL/models/PPO/du/PPO_3"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def linear_decay_schedule(initial_value: float, final_value:float) -> Callable[[float], float]:
    def schedule(progress: float) -> float:
        # Linearly decay the learning rate from `initial_value` to 0
        return initial_value - progress*(initial_value- final_value)
    return schedule

learning_rate_schedule = get_schedule_fn(linear_decay_schedule(1e-5, 1e-4))

# Instantiate and train the PPO model as per the Du paper
model = PPO(
    policy= Du_ActorCriticPolicy, 
    env= vec_env, 
    learning_rate=learning_rate_schedule,
    n_steps= 1500, #update every 15000 episodes
    n_epochs = 5, 
    clip_range =0.2, 
clip_range_vf = 0.2,
    verbose=1,
    gae_lambda=0.95, 
    gamma=0.9,
    ent_coef=0.1,  # c2
    vf_coef=0.3,  #c1
   normalize_advantage= True,
    tensorboard_log=log_dir,
    device= "auto"
)


# Train the model in increments and save after each block of timesteps
TIMESTEPS = 1000000
for i in range(1,11):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_3")
    model.save(f"{models_dir}/ppo_{TIMESTEPS*i}")



