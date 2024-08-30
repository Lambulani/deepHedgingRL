import gym
import numpy as np
from gym import Env
from gym import error, spaces, utils
from gym.utils import seeding



# ### Setting up the Environment

#Setting up the action space for DQN 
action_mapping = np.arange(-100, 101) 
#define the hedging object 
class env_hedging(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, asset_price_model, dt, T, num_steps=100, cost_multiplier = 0, tick_size=0.01,
                 L=1, strike_price=None, int_holdings=True, initial_holding=0, mode="PL", **kwargs):
        super().__init__()
        self.action_space = spaces.Discrete(201)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        assert (mode in ["PL", "CF"]), "Only 'PL' and 'CL' are allowed values for mode."
        self.asset_price_model = asset_price_model
        self.current_price = asset_price_model.get_current_price()
        self.n = 0
        self.T = T
        self.dt = dt
        if mode == "PL":
            self.option_price_model = kwargs.get('option_price_model', None)
            assert (self.option_price_model is not None), "If 'PL' is chosen a option_price_model needs be be provided."
            self.current_option_price = self.option_price_model.compute_option_price(self.n, self.current_price)
            self.delta = self.option_price_model.compute_delta(0, self.current_price)
        self.done = False
        self.num_steps = num_steps
        self.mode = mode
        self.L = L
        self.initial_holding = initial_holding
        self.h = initial_holding
        self.int_holdings = int_holdings
        if int_holdings:
            self.h = round(self.h)
        if strike_price:
            self.strike_price = strike_price
        else:
            self.strike_price = self.current_price
        self.tick_size = tick_size
        self.cost_multiplier = cost_multiplier
    def _compute_cf_reward(self, new_h, next_price, delta_h):
        reward = -self.current_price * delta_h - self.cost_multiplier * self.tick_size* np.abs(self.current_price * delta_h)
        if self.done:
            asset_value = next_price * new_h - self.cost_multiplier *self.tick_size * np.abs(next_price * new_h)
            payoff = self.L *100* max(0, next_price - self.strike_price)
            reward += asset_value - payoff
        return reward 
    def _compute_pl_reward(self, new_h, next_price, delta_h):
        new_option_price = self.option_price_model.compute_option_price(self.n, next_price)
        reward_option_price = self.L * (100*(new_option_price - self.current_option_price) - new_h*
                                        (next_price - self.current_price))
        reward_trading_cost = - self.cost_multiplier* self.tick_size * self.dt * (abs(delta_h) + 0.01 * delta_h**2)
        delta_wealth = reward_option_price + reward_trading_cost
        reward = delta_wealth - (0.1/2)*(delta_wealth**2) #reward function according to Kolm (2019)
        self.current_option_price = new_option_price
        if self.done:
            pass
        return reward

    def step(self, delta_h):
        delta_h = action_mapping[delta_h]
        if self.int_holdings:
            delta_h = round(delta_h)
        new_h = self.h + delta_h
        self.asset_price_model.compute_next_price()
        next_price = self.asset_price_model.get_current_price()
        self.n += 1
        if self.n == self.num_steps:
            self.done = True

        if self.mode == "CF":
            reward = self._compute_cf_reward(new_h, next_price, delta_h)
        elif self.mode == "PL":
            reward = self._compute_pl_reward(new_h, next_price, delta_h)
            self.delta = self.option_price_model.compute_delta(self.n, self.current_price)
        else:
            assert "error 1"
        self.current_price = next_price
        self.h = new_h
        state = self.get_state()
        info = {}
        return [state, reward, self.done, self.done,  info]

    def get_state(self):
        time_to_maturity = self.T - self.n * self.dt
        if self.mode == "PL":
            return np.array([self.h, self.current_price, time_to_maturity, self.current_option_price, self.delta],
                            dtype=float)
        else:
            return np.array([self.h, self.current_price, time_to_maturity], dtype=float)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
            
        self.asset_price_model.reset()
        self.n = 0
        self.done = False
        self.current_price = self.asset_price_model.get_current_price()
        self.h = self.initial_holding
        if self.mode == "PL":
            self.current_option_price = self.option_price_model.compute_option_price(self.n, self.current_price)
            self.delta = self.option_price_model.compute_delta(self.n, self.current_price)
        if self.int_holdings:
            self.h = round(self.h)
        state = self.get_state()
        info = {}
        
        return state, info

    def render(self, mode='human', close=False):
        return self.get_state()







