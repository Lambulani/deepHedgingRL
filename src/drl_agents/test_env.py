import gymnasium as gym
import numpy as np
from src.pricing.asset_price_process import GBM
from src.pricing.option_price_process import BSM
from src.custom_environments.HedgeEnv import env_hedging

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
                     L=1, strike_price=strike_price, integer_holdings=True, initial_holding=0, mode="PL", act_space_type= "continuous",shares_per_contract=100,
                  option_price_model=opm)

def main():
    env.step(0.2)
    print(env.get_state())
if __name__ == "__main__":
    main()


