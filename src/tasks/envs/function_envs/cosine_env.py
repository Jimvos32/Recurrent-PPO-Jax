from src.tasks.envs.base_fun_env import BaseOptimizationEnv
import numpy as np

class CosineEnv(BaseOptimizationEnv):
    """
    Environment based on cosine functions with additional small oscillations.
    """
    def __init__(self, env_config=None):
        
        self.num_oscillations = env_config["num_oscillations"]
        self.c_bounds = env_config.get("c_bounds", (5.0, 20.0))
        self.A_bounds = env_config.get("A_bounds", (10.0, 20.0))
        self.B_bounds = env_config.get("B_bounds", (0.5, 1.5))
        self.small_A_bounds = env_config.get("small_A_bounds", (0.2, 3.0))
        self.small_B_bounds = env_config.get("small_B_bounds", (0.5, 4.0))
        super().__init__(env_config)

    def initialize_function(self):
        
        self.c = np.random.uniform(self.c_bounds[0], self.c_bounds[1])
        self.A0 = np.random.uniform(self.A_bounds[0], self.A_bounds[1])
        self.B0 = np.random.uniform(self.B_bounds[0], self.B_bounds[1], size=(self.action_dim,))
        self.s0 = np.random.uniform(self.x_range[0] * 0.9, self.x_range[1] * 0.9, size=(self.action_dim,))
        
        self.small_A = np.random.uniform(self.small_A_bounds[0], self.small_A_bounds[1], size=(self.action_dim, self.num_oscillations))
        self.small_B = np.random.uniform(self.small_B_bounds[0], self.small_B_bounds[1], size=(self.action_dim, self.num_oscillations))
        
        self.small_shift = np.tile(self.s0.reshape(-1, 1), (1, self.num_oscillations))
        self.small_phase = np.zeros((self.action_dim, self.num_oscillations))
        # In this cosine environment the optimum is at s0.
        self.optimum_point = self.s0
        # Compute theoretical bounds.
        self.max_y = self.c + self.A0 * self.action_dim + np.sum(self.small_A)
        self.min_y = self.c - self.A0 * self.action_dim - np.sum(self.small_A)
        

    def compute_y(self, x):
        x = np.atleast_2d(x)
        result = np.full(x.shape[0], self.c)
        # Add the large cosine peak for each dimension.
        for i in range(self.action_dim):
            result += self.A0 * np.cos(self.B0[i] * (x[:, i] - self.s0[i]))
        # Add the small oscillations.
        for i in range(self.action_dim):
            for k in range(self.num_oscillations):
                diff = x[:, i] - self.small_shift[i, k]
                result += self.small_A[i, k] * np.cos(self.small_B[i, k] * diff + self.small_phase[i, k])
        return result.squeeze()
