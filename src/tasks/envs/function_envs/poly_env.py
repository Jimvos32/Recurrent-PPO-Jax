from src.tasks.envs.base_fun_env import BaseOptimizationEnv
import numpy as np

class PolyEnv(BaseOptimizationEnv):
    """
    Environment based on a polynomial function with a multidimensional mask.
    """
    def __init__(self, env_config=None):
        self.degree = env_config.get("degree", 2)
        self.c_bounds = env_config.get("c_bounds", (5.0, 20.0))
        self.weight_bounds = env_config.get("x_bounds", (0.5, 2.0))
        super().__init__(env_config)

    def initialize_function(self):
        # Set polynomial-specific parameters.
        self.x_max = np.random.uniform(self.x_range[0] * 0.9, self.x_range[1] * 0.9, size=(1,self.action_dim))
        self.c = np.random.uniform(self.c_bounds[0], self.c_bounds[1])
        self.weights = np.random.uniform(self.weight_bounds[0], self.weight_bounds[1], size=(self.action_dim,))
        # The optimum is at x_max.
        self.optimum_point = self.x_max
        
        self.min_y = self.find_minimum()

    def find_minimum(self):
        lower, upper = self.x_range
        # For each dimension, choose the bound that is farther from x_max.
        x_min = np.where((self.x_max - lower) > (upper - self.x_max), lower, upper)
        y_min = self.compute_y(x_min)
        return y_min

    def compute_y(self, x):
        diff = x - self.x_max
        weighted_term = self.weights * (diff ** self.degree)
        result = self.c - np.sum(weighted_term, axis=1)
        return result.squeeze()
