from ..feature_generator import FeatureGenerator
from ..feature_sets import EmaFeatureSet
from ..agent_base import AgentBase
import numpy as np

WINDOW = 390
MUL = 1

class RollingBands(AgentBase):

    def __init__(self, name, stock_list, window=WINDOW, deviation_multiplier=MUL):
        self.name = name
        self.feature_set = EmaFeatureSet((window,))
        self.stock_list = stock_list
        self.window = window
        self.deviation_multiplier = deviation_multiplier
        self.reset()

    def reset(self, eval_stock_list=None):
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        for stock in stock_list:
            self.feature_set.append_initial_prices(stock.get_prices(1000))

    def take_action(self, eval_stock_list=None):
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        action_set = dict()
        for stock in stock_list:
            action = self.take_action_for_stock(stock)
            action_set[stock.ticker] = action / len(stock_list)

        return action_set

    def take_action_for_stock(self, stock):
        price = stock.price()

        self.feature_set.append_price(price)

        moving_avg = self.feature_set.mean[0]
        deviation = np.sqrt(self.feature_set.var[0]) * self.deviation_multiplier

        return min(1, max(0, (moving_avg + deviation - price) / (2 * deviation)))



