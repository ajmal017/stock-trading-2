from . import FeatureSetBase

import numpy as np

SCALING_FACTOR = 0.1

class EmaFeatureSet(FeatureSetBase):

    def __init__(self, moving_windows):
        self.length = len(moving_windows)
        self.alpha = 2 / (np.array(moving_windows) + 1)

    def append_initial_prices(self, prices):
        self.prev_price = 0
        self.price = prices[0]
        self.mean = prices[0] * np.ones(self.length)
        self.var = np.ones(self.length)
        for p in prices[1:]:
            self.append_price(p)

    def append_price(self, price):
        self.prev_price = self.price
        self.price = price
        diff = price - self.mean
        self.mean += self.alpha * diff
        self.var += self.alpha * diff * diff
        self.var *= (1 - self.alpha)

    def get_features(self):
        return self.get_normalized_value(self.price)

    def get_normalized_value(self, value):
        if min(self.var) == 0:
            return (value - self.mean) / (np.sqrt(self.var) + 1)
        return SCALING_FACTOR * (value - self.mean) / np.sqrt(self.var)

    def __len__(self):
        return self.length

    def get_min_init_price_length(self):
        return 1
