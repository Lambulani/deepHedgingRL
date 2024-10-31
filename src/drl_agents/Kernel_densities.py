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

def cost(delta_h, multiplier):
    TickSize = 0.1
    return multiplier * TickSize * (np.abs(delta_h) + 0.01 * delta_h**2)


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
from torch.nn import functional as F
import os 

class Du_Custom_Network(nn.Module):
    
    def __init__(self, feature_dim: int = 5,last_layer_dim_pi: int = 201,last_layer_dim_vf: int = 1, hidden_dim: int = 90):
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


custom_objects = {"policy_class": Du_ActorCriticPolicy}
model = PPO.load("/home/bndlev001/deepHedgingRL/models/PPO/du/PPO_7/ppo_49000000", custom_objects= custom_objects)

def student_t_statistic(data):
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    t_statistic = sample_mean / (sample_std / np.sqrt(n))
    return t_statistic

# Getting kernel density estimates for cost and volatility
num_episodes = 30000
cost_pnls_dh = []
cost_pnls_ppo = []
total_pnls_vol_dh = []
total_pnls_vol_ppo = []
t_stat_pnls_dh = []
t_stat_pnls_ppo = []

for episode in range(num_episodes): 
    state = env.reset()
    done = False
    samp_cost_pnl_dh = 0
    samp_cost_pnl_ppo = 0

    pnl_diffs_ppo = []
    pnl_diffs_dh = []
    
    previous_pnl_ppo = 0
    previous_pnl_dh = 0
    
    total_pnl_ppo = 0
    total_pnl_dh = 0
    samp_total_pnl_dh = []
    samp_total_pnl_ppo = []

    while not done:
        current_holdings, current_asset_price, current_ttm, current_option_value, current_delta= env.get_state()

        action, _states = model.predict(env.get_state(), deterministic=True)
        next_state, reward, done,done,  info = env.step(action)
        next_holdings, next_asset_price, next_ttm, next_option_value, next_delta = next_state
        
        delta_h_ppo = next_holdings-current_holdings
        delta_h_dh = (-round(next_delta, 2)) - (-round(current_delta, 2))


        # Cost calculations for both policies
        samp_cost_pnl_dh += cost(delta_h_dh, 5)
        samp_cost_pnl_ppo += cost(delta_h_ppo, 5)

        # P&L differences between steps
        pnl_ppo = ((next_option_value - current_option_value) - current_holdings * (next_asset_price - current_asset_price) - cost(delta_h_ppo, 5))
        pnl_dh = ((next_option_value - current_option_value) + (-round(current_delta, 2)) * (next_asset_price - current_asset_price) - cost(delta_h_dh, 5))
        
        pnl_diffs_ppo.append(pnl_ppo - previous_pnl_ppo)
        pnl_diffs_dh.append(pnl_dh - previous_pnl_dh)

        # Update previous P&L
        previous_pnl_ppo = pnl_ppo
        previous_pnl_dh = pnl_dh

        # Accumulate the cumulative P&L for each policy
        total_pnl_ppo+= pnl_ppo
        total_pnl_dh += pnl_dh
        samp_total_pnl_ppo.append(total_pnl_ppo)
        samp_total_pnl_dh.append(total_pnl_dh)



        if done:
            # Volatility is the standard deviation of P&L differences
            vol_ppo = np.std(pnl_diffs_ppo)
            vol_dh = np.std(pnl_diffs_dh)

            # Calculate student t-statistic based on cumulative P&L at the end of the episode
            t_stat_pnl_ppo = student_t_statistic(samp_total_pnl_ppo)  # Use cumulative P&L
            t_stat_pnl_dh = student_t_statistic(samp_total_pnl_dh)  # Use cumulative P&L

            # Append cost and volatility for this episode
            cost_pnls_ppo.append(samp_cost_pnl_ppo)
            cost_pnls_dh.append(samp_cost_pnl_dh)
            total_pnls_vol_ppo.append(vol_ppo)
            total_pnls_vol_dh.append(vol_dh)
            t_stat_pnls_ppo.append(t_stat_pnl_ppo)
            t_stat_pnls_dh.append(t_stat_pnl_dh)

            state = env.reset()

# Plotting the KDE and results
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Plot kernel density estimates for total cost
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
sns.kdeplot(cost_pnls_ppo, label='Policy: ppo', shade=True)
sns.kdeplot(cost_pnls_dh, label='Policy: $\delta_{DH}$', shade=True)
plt.title('KDE for Total Cost')
plt.xlabel('Total Cost')
plt.ylabel('Density')
plt.legend()

# Plot kernel density estimates for volatility of total P&L
plt.subplot(1, 3, 2)
sns.kdeplot( total_pnls_vol_ppo, label='Policy: ppo', shade=True)
sns.kdeplot(total_pnls_vol_dh, label='Policy: $\delta_{DH}$', shade=True)
plt.title('KDE for Volatility of Total P&L')
plt.xlabel('Volatility of Total P&L')
plt.ylabel('Density')
plt.legend()

# Plot KDE for Student's t-statistic of Cumulative P&L
plt.subplot(1, 3, 3)
sns.kdeplot(t_stat_pnls_ppo, label='Policy: ppo', shade=True)
sns.kdeplot(t_stat_pnls_dh, label='Policy: $\delta_{DH}$', shade=True)
plt.title("Student's t-Statistic KDE for Cumulative P&L")
plt.xlabel('Student t-Statistic')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.savefig("PPO_Kernel_Densities_38500")
plt.show()
