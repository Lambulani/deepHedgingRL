import sys 
sys.path.append("C:/Users/levyb/Documents/Masters Data Science - 2nd Year/deepHedgingRL")
import gym
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
seed =2024


apm = GBM(mu=mu, dt=dt, s_0=s_0, sigma=sigma)
opm = BSM(strike_price=strike_price, risk_free_interest_rate=r, volatility=sigma, T=T, dt=dt)
env = env_hedging_ppo(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0.005, tick_size=0.01,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL",
                  option_price_model=opm, seed = seed)



from stable_baselines3 import PPO
from gym import spaces
from typing import Callable, Tuple
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_schedule_fn
import os 


n_envs = 2
vec_env = make_vec_env(lambda:env_hedging_ppo(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0.005, tick_size=0.01,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL",
                  option_price_model=opm) , n_envs= n_envs, seed = seed)



class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function with 5 hidden layers.
    Each layer uses ReLU activation, witorch batch normalization applied before ReLU.
    
    :param feature_dim: dimension of torche features extracted by the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 201,
        last_layer_dim_vf: int = 1,
        hidden_dim: int = 64,  # Hidden layer dimension
    ):
        super().__init__()

        # Save output dimensions, used to create the distributions
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
            nn.Linear(hidden_dim, last_layer_dim_pi),  # Final layer
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
            nn.Linear(hidden_dim, last_layer_dim_vf),  # Final layer
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
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
        self.mlp_extractor = CustomNetwork(self.features_dim)


# Custom callback to log rewards during training

class RewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(RewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.total_reward = 0.0
        self.episode_count = 0

    def _on_training_start(self) -> None:
        # Open file for writing average rewards at the start of training
        self.file = open(os.path.join(self.log_dir, 'avg_rewards_ppo.txt'), 'w')

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]  # Access the current reward
        self.total_reward += reward
        self.episode_count += 1

        # Log rewards at each check frequency
        if self.n_calls % self.check_freq == 0:
            # Calculate average reward
            avg_reward = (self.total_reward / self.episode_count)/n_envs
            timestep = self.n_calls * self.locals['env'].num_envs  # Total timesteps processed
            self.file.write(f"{timestep}, {avg_reward}\n")
            self.file.flush()  # Ensure it's written immediately
        return True

    def _on_training_end(self) -> None:
        # Close the file after training ends
        self.file.close()

#Have a learning rate schedule over training. 
def linear_decay_schedule(initial_value: float, final_value:float) -> Callable[[float], float]:
    def schedule(progress: float) -> float:
        # Linearly decay the learning rate from `initial_value` to 0
        return initial_value - progress*(initial_value- final_value)
    return schedule

# Create a linear decay schedule with initial learning rate 1e-4
learning_rate_schedule = get_schedule_fn(linear_decay_schedule(1e-4, 1e-5))

# Set up callback and directories for logging and saving models
log_dir = "C:/Users/levyb/Documents/Masters Data Science - 2nd Year/deepHedgingRL/results/PPO/du/logs"
callback = RewardCallback(check_freq=1000, log_dir=log_dir, verbose=1)

models_dir = "C:/Users/levyb/Documents/Masters Data Science - 2nd Year/deepHedgingRL/models/PPO/du/PPO_6"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)



# Instantiate and train the PPO model as per the Du paper
model = PPO(
    policy= CustomActorCriticPolicy, 
    env= vec_env, 
    learning_rate=learning_rate_schedule,
    n_steps= 7500, 
    n_epochs = 5, 
    clip_range =0.2, 
    verbose=1,
    gae_lambda=0.9, 
    gamma=0.9,
    ent_coef=0.2,  # Entropy coefficient for exploration c2 parameter
    vf_coef=0.5,  # Value function coefficient c1 parameter
    clip_range_vf= 0.2, 
    normalize_advantage= True, 
    tensorboard_log=log_dir,
)


# Train the model in increments and save after each block of timesteps
TIMESTEPS = 100000
for i in range(1, 60):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_6", callback=callback, progress_bar= True)
    model.save(f"{models_dir}/ppo_{TIMESTEPS*i}")



