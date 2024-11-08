#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from abc import ABC, abstractmethod


# In[2]:


class GenericAssetPriceModel(ABC):
    @abstractmethod
    def get_current_price(self):
        pass

    @abstractmethod
    def compute_next_price(self, *action):
        pass

    @abstractmethod
    def reset(self):
        pass


# In[3]:


class GBM(GenericAssetPriceModel):
    def __init__(self, mu=0, dt=1/5, s_0=100, sigma=0.01):
        self.mu = mu
        self.dt = dt
        self.s_0 = s_0
        self.sigma = sigma
        self.current_price = s_0

    def compute_next_price(self):
        # Wiener process increment
        dz = self.rng.normal(0, 1) * np.sqrt(self.dt)
        
        # GBM formula
        new_price = self.current_price * np.exp((self.mu - 0.5 * self.sigma ** 2) * self.dt
                   + self.sigma * dz)
        self.current_price = new_price
    
    def seed(self, seed=None):
        # Set the seed for the random generator
        self.rng = np.random.default_rng(seed)

    def reset(self):
        # Reset to initial price
        self.current_price = self.s_0

    def get_current_price(self):
        # Return the current price
        return self.current_price


# In[ ]:




