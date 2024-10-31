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

env = env_hedging(asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, cost_multiplier = 0, tick_size=0.01,
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


# Set up directories for logging and saving models
models_dir = "/home/bndlev001/deepHedgingRL/models/PPO/du/grid_search_simple"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Objective function for Optuna
def objective(trial):
    # Suggest different hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.1, 0.2)
    vf_coef = trial.suggest_loguniform('vf_coef', 0.1, 1.0)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    gamma =  trial.suggest_uniform('gamma', 0.85, 0.90)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.95, 0.99)
    clip_range_vf = trial.suggest_uniform('clip_range_vf', 0.1, 0.3)
    num_hidden_layers =trial.suggest_int('num_hidden_layers', 3, 5)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)  # Adjust hidden layer size

    policy_kwargs = {"activation_fn": nn.ReLU, 
                     "net_arch": [hidden_dim]* num_hidden_layers}


    # Instantiate the PPO model with the sampled hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs= policy_kwargs
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        clip_range=clip_range,
        gamma= gamma,
        gae_lambda= gae_lambda,
        clip_range_vf= clip_range_vf,
        normalize_advantage= True, 
        tensorboard_log=models_dir,
        n_steps=75000,
	batch_size = 5000,
        n_epochs=6,
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

