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
from stable_baselines3.common.callbacks import EvalCallback
from torch.nn import functional as F
import os 
import optuna
import pickle

n_envs = 10
vec_env = make_vec_env(lambda:env_hedging_ppo(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.01,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL",
                  option_price_model=opm) , n_envs= n_envs)



class Du_Custom_Network(nn.Module):
    
    def __init__(self, feature_dim: int = 5,last_layer_dim_pi: int = 201,last_layer_dim_vf: int = 1, hidden_dim: int = 64):
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
models_dir = "/home/bndlev001/deepHedgingRL/models/PPO/du/grid_search"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Objective function for Optuna
def objective(trial):
    # Suggest different hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-2, 0.2)
    vf_coef = trial.suggest_loguniform('vf_coef', 0.1, 1.0)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    gamma =  trial.suggest_uniform('gamma', 0.85, 0.90)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.95, 0.99)
    clip_range_vf = trial.suggest_uniform('clip_range_vf', 0.1, 0.3)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)  # Adjust hidden layer size

    # Create a new policy with the sampled hyperparameters
    class CustomPolicy(ActorCriticPolicy):
        def _build_mlp_extractor(self):
            self.mlp_extractor = Du_Custom_Network(self.features_dim, hidden_dim=hidden_dim)

    # Instantiate the PPO model with the sampled hyperparameters
    model = PPO(
        policy=CustomPolicy,
        env=vec_env,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        clip_range=clip_range,
        gamma= gamma
        gae_lambda= gae_lambda
        clip_range_vf= clip_range_vf,
        normalize_advantage= True, 
        tensorboard_log=models_dir,
        n_steps=1500,
        n_epochs=5,
        verbose=1,
        device="auto"
    )
    
    # Create an evaluation callback to monitor performance
    eval_env = env_hedging_ppo(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.01,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL",
                  option_price_model=opm)
    eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir, log_path=models_dir, eval_freq=5000)
    
    # Train the model
    model.learn(total_timesteps=3000000, callback=eval_callback)

    # Get the last mean reward from the evaluation log
    mean_reward = eval_callback.last_mean_reward
    return mean_reward

# Create an Optuna study for hyperparameter optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials= 20)

# Output the best hyperparameters
print("Best hyperparameters:", study.best_params)

# Save the study for future reference
with open(os.path.join(models_dir, 'study.pkl'), 'wb') as f:
    pickle.dump(study, f)

