# import sys 
# sys.path.append("/home/bndlev001/deepHedgingRL")
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
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL", kappa = 0.1, act_space_type="discrete", shares_per_contract=100,
                  option_price_model=opm)




from stable_baselines3 import PPO
from gym import spaces
from typing import Callable, Tuple
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import functional as F
import os 


n_envs = 4
vec_env = make_vec_env(lambda:env_hedging(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.1,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL",kappa=0.1,  act_space_type= "discrete", shares_per_contract=100,
                  option_price_model=opm), n_envs= n_envs)


# Set up directories for logging and saving models
log_dir = "results/PPO/du/logs"

models_dir = "models/PPO/du/PPO_48"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def linear_decay_schedule(initial_value: float, final_value:float) -> Callable[[float], float]:
    def schedule(progress: float) -> float:
        # Linearly decay the learning rate from `initial_value` to 0
        return initial_value - progress*(initial_value- final_value)
    return schedule

learning_rate_schedule = get_schedule_fn(linear_decay_schedule(1e-4, 1e-5))


class DuPPONetwork(nn.Module):
    def __init__(self, feature_dim:int, last_layer_dim_pi : int =256, 
                 last_layer_dim_vf: int =256):
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        layers = []
        layer_size = 256
        for _ in range(5):  # 5 hidden layers
            layers.append(nn.Linear(feature_dim, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
            feature_dim = layer_size  # Update input dim for the next layer
        

        self.policy_net = nn.Sequential(*layers)
        self.value_net = nn.Sequential(*layers)

    def forward(self, features: th.Tensor)-> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features: th.Tensor)-> th.Tensor:
        return self.policy_net(features)
    
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

class DuActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self, observation_space: spaces.Box,
        action_space: spaces.Discrete,
        lr_schedule: Callable[[float], float],
        *args, 
        **kwargs
    ):
        kwargs["ortho_init"]= True
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DuPPONetwork(self.features_dim)
    
policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch = dict(pi = [64,64,64,64,64],vf = [64,64,64,64,64]))

# Instantiate and train the PPO model as per the Du paper
model = PPO(
    policy= DuActorCriticPolicy, 
    env= vec_env, 
    learning_rate=learning_rate_schedule,
    n_steps= 3750, #update every 750,000 episodes
    n_epochs = 5,
batch_size = 1000, 
    clip_range= 0.2, 
clip_range_vf = 0.2,
    verbose=1,
    gae_lambda= 0.9871191601388827, 
    gamma= 0.9,
    ent_coef=0.2,  # c2
    vf_coef=0.5,  #c1
    max_grad_norm= 0.5, 
   normalize_advantage= True,
    tensorboard_log=log_dir,
    device= "auto"
)


# Train the model in increments and save after each block of timesteps
TIMESTEPS = 100000
for i in range(1,200):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_48", progress_bar= True )
    model.save(f"{models_dir}/ppo_{TIMESTEPS*i}")
    



