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
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL", kappa = 0.1, act_space_type= "continuous",shares_per_contract=100,
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
vec_env = make_vec_env(lambda:env_hedging(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.1,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL",kappa=0.1,  act_space_type= "continuous", shares_per_contract=100,
                  option_price_model=opm), n_envs= n_envs)


class DuPPONetwork(nn.Module):
    def __init__(self, feature_dim:int, last_layer_dim_pi : int =256, 
                 last_layer_dim_vf: int = 256):
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


# Set up directories for logging and saving models
models_dir = "/home/bndlev001/deepHedgingRL/models/PPO/du/grid_search_1"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Objective function for Optuna
def objective(trial):
    # Suggest different hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-4)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-2, 0.2)
    vf_coef = trial.suggest_loguniform('vf_coef', 0.5, 1.0)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    gamma =  trial.suggest_uniform('gamma', 0.8, 1)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.95, 0.99)
    clip_range_vf = trial.suggest_uniform('clip_range_vf', 0.1, 0.3)
    
    

    # Instantiate the PPO model with the sampled hyperparameters
    model = PPO(
        policy= DuActorCriticPolicy,
        env=vec_env,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        clip_range=clip_range,
        gamma= gamma,
        gae_lambda= gae_lambda,
        clip_range_vf= clip_range_vf,
        normalize_advantage= True, 
        tensorboard_log=models_dir,
        n_steps=1500,
        n_epochs=5,
        verbose=1,
        device="auto"
    )
    
    # Create an evaluation callback to monitor performance
    eval_env = env_hedging(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.1,
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL", kappa = 0.1, act_space_type= "continuous",shares_per_contract=100,
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

