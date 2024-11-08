import gymnasium as gym

import numpy as np
from gym import Env, spaces
from gym.utils import seeding



# ### Setting up the Environment 
#define the hedging object 
class env_hedging(Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, asset_price_model, dt, T, num_steps= 50, cost_multiplier = 0, tick_size=0.1,
                 L=1, strike_price=100, integer_holdings =True, initial_holding=0, mode="PL", kappa = 0.1, shares_per_contract =100, act_space_type= "discrete", **kwargs):
        super().__init__()
        self.act_space_type = act_space_type
        assert (self.act_space_type in  ["discrete", "continuous"])
        if act_space_type == "discrete":
            self.action_space = spaces.Discrete(201)
        else:
            self.action_space = spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        assert (mode in ["PL", "CF"]), "Only 'PL' and 'CL' are allowed values for mode."
        self.kappa = kappa 
        self.prev_delta_wealth = 0
        self.asset_price_model = asset_price_model
        self.current_price = asset_price_model.get_current_price()
        self.n = -1
        self.T = T
        self.dt = dt
        self.shares_per_contract = shares_per_contract
        if mode == "PL":
            self.option_price_model = kwargs.get('option_price_model', None)
            assert (self.option_price_model is not None), "If 'PL' is chosen a option_price_model needs be be provided."
            self.current_option_price = self.option_price_model.compute_option_price(0, self.current_price)
            self.delta = self.option_price_model.compute_delta(0, self.current_price)
        self.done = False
        self.num_steps = num_steps
        self.mode = mode
        self.L = L
        self.initial_holding = -100*self.delta
        self.h = initial_holding
        self.integer_holdings  = integer_holdings 
        self.delta_mapping = np.arange(-100,101)
        if integer_holdings :
            self.h = round(self.h)
        if strike_price:
            self.strike_price = strike_price
        else:
            self.strike_price = self.current_price
        self.tick_size = tick_size
        self.cost_multiplier = cost_multiplier
    # def _compute_cf_reward(self, new_h, next_price, delta_h):
    #     reward = -self.current_price * delta_h - self.cost_multiplier * self.tick_size* np.abs(self.current_price * delta_h)
    #     if self.done:
    #         asset_value = next_price * new_h - self.cost_multiplier *self.tick_size * np.abs(next_price * new_h)
    #         payoff = self.L *100* max(0, next_price - self.strike_price)
    #         reward += asset_value - payoff
    #     return reward
    
    def _compute_pl_reward(self, prev_h, current_price, next_price, delta_h):
        if self.n ==0 :
            reward =-self.cost_multiplier* self.tick_size * (abs(prev_h) + 0.01 * prev_h**2)
            return reward
        new_option_price = self.option_price_model.compute_option_price(self.n, next_price)
        delta_V = self.L * self.shares_per_contract*(new_option_price - self.current_option_price)
        delta_S = prev_h*(next_price - current_price)
        trading_cost =  self.cost_multiplier* self.tick_size * (abs(delta_h) + 0.01 * delta_h**2)
        delta_wealth = delta_V - delta_S - trading_cost 
        self.current_option_price = new_option_price
        
        if self.done:
            # asset_value = next_price * (prev_h + delta_h) 
            termination_cost = self.cost_multiplier* self.tick_size * (abs(prev_h + delta_h) + 0.01 * (prev_h + delta_h)**2)
            payoff = self.L *self.shares_per_contract* max(0, next_price - self.strike_price)
            delta_wealth +=  payoff - termination_cost
        
        delta_wealth +=1/self.kappa
        norm_factor = 1/1e4
        reward = (delta_wealth - (self.kappa/2)*(abs(delta_wealth)**2)) #reward function according to Kolm (2019), quadratic utility mean variance optimization 
        reward = norm_factor*reward

        tolerance = 0.01
        if abs(delta_wealth) < tolerance:
            reward += 0.1  # Incentivize close-to-zero PnL
        
        if delta_wealth > self.prev_delta_wealth:
            reward += 0.1 # Reward reduction in negative PnL
        self.prev_delta_wealth = delta_wealth

        return reward

    def step(self, delta_h):
        if self.act_space_type == "discrete":
            delta_h = self.delta_mapping[delta_h]
        else: 
            delta_h = delta_h[0]

        new_h = self.h + delta_h
        #if abs(new_h) > self.shares_per_contract:
        #    new_h = new_h - delta_h
        #    delta_h = 0
        current_price=self.asset_price_model.get_current_price()
        self.asset_price_model.compute_next_price()
        next_price = self.asset_price_model.get_current_price()
        self.n += 1
        if self.n == self.num_steps:
            self.done = True

        if self.mode == "CF":
            reward = self._compute_cf_reward(new_h, next_price, delta_h)
        elif self.mode == "PL":
            reward = self._compute_pl_reward(self.h, current_price, next_price, delta_h)
            self.delta = self.option_price_model.compute_delta(self.n, self.current_price)
        else:
            assert "error 1"

        self.current_price = next_price
        self.h = new_h
        state = self.get_state()
        info = {}
        return state, reward, self.done,  self.done,  info

    def get_state(self):
        time_to_maturity = self.T - self.n * self.dt
        if self.mode == "PL":
            return np.array([self.h, self.current_price, time_to_maturity, self.current_option_price])
        else:
            return np.array([self.h, self.current_price, time_to_maturity])
        
    def set_state(self, h, current_price, time_to_maturity, current_option_price, delta):
        self.h = h
        self.current_price= current_price
        self.n= int((self.T-time_to_maturity)/self.dt)
        self.current_option_price = current_option_price
        self.delta= delta 
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**31-1)

        self.np_random, seed = seeding.np_random(seed)
        return seed


    def reset(self, seed = None, options = None):
        super().reset(seed = seed) 
        self.asset_price_model.seed(seed)
        self.asset_price_model.reset()
        self.n = 0
        self.done = False
        self.current_price = self.asset_price_model.get_current_price()
        self.h = self.initial_holding
        if self.mode == "PL":
            self.current_option_price = self.option_price_model.compute_option_price(self.n, self.current_price)
            self.delta = self.option_price_model.compute_delta(self.n, self.current_price)
        if self.integer_holdings :
            self.h = round(self.h)
        state = self.get_state()
        info = {}
        return state, info

    def render(self, mode='human', close=False):
        return self.get_state()



