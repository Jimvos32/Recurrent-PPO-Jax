import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp



class MultivariatePolyEnv(gym.Env):
    """
    A Gymnasium environment where the agent interacts with a multivariate polynomial function.
    The agent selects an action (an n-dimensional vector) and the observation is a scalar, the value
    of the function. The function is defined as:
    
      f(x) = c - sum_{i=1}^{action_dim} (w_i * (x_i - x_max_i)**degree)
    
    where:
      - degree: the degree of the polynomial (for unimodality, degree should be even).
      - action_dim: the dimensionality of the input vector.
      - x_max: the location of the global maximum (randomized at reset).
      - c: a constant such that f(x_max) = c.
      - w_i: positive weights for each dimension.
    
    The reward is computed as the negative absolute difference between f(x) and the maximum value f(x_max)=c.
    """
    def __init__(self, env_config=None, degree=2, action_dim=2, x_range=(-10, 10), max_episode_steps=12):
        super(MultivariatePolyEnv, self).__init__()
        self.degree = degree
        self.action_dim = action_dim
        self.x_range = x_range  # same range for each dimension
        self.max_episode_steps = max_episode_steps
        
        # Define the action space: a continuous vector of length 'action_dim'
        low = np.full((action_dim,), x_range[0], dtype=np.float32)
        high = np.full((action_dim,), x_range[1], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Observation space: a scalar value representing f(x)
        self.observation_space = spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(1,), dtype=jnp.float32)
        
        self.tick = 0
        self.raw_rewards = []
        self.resetted = 0
        
        self.state = None
        self.x = None
        
        # Polynomial parameters (to be randomized at each reset)
        self.x_max = None      # the location of the maximum (vector)
        self.c = None          # constant shift (f(x_max) = c)
        self.weights = None    # positive weights for each dimension
        
        # Initialize the polynomial parameters
        self.shift_polynomial()
    
    # def shift_polynomial(self):
    #     """
    #     Randomize the polynomial parameters to create a new function.
    #     For unimodality, degree should be even. If an odd degree is provided, the function is still generated
    #     but unimodality is not guaranteed.
    #     """
    #     # Randomize the maximum location x_max uniformly for each dimension.
    #     self.x_max = np.random.uniform(self.x_range[0], self.x_range[1], size=(self.action_dim,))
        
    #     # Randomize constant shift and weights.
    #     self.c = np.random.uniform(5.0, 20.0)
    #     self.weights = np.random.uniform(-0.5, -2.0, size=(self.action_dim,))
    
    
    def shift_polynomial(self):
        # Randomize the maximum location x_max uniformly for each dimension
        self.x_max = np.random.uniform(self.x_range[0], self.x_range[1], size=(self.action_dim,))
        
        # Randomize c (the peak value) and weights (must be positive)
        self.c = np.random.uniform(5.0, 20.0)
        self.weights = np.random.uniform(0.5, 2.0, size=(self.action_dim,))

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize the polynomial parameters for a new episode.
        self.shift_polynomial()
        
        # Start with a random x value within the allowed range.
        self.x = np.random.uniform(self.x_range[0], self.x_range[1], size=(self.action_dim,))
        y = self.compute_y(self.x)
        self.state = jnp.array([y], dtype=jnp.float32)
        
        self.tick = 0
        self.raw_rewards = []
        self.resetted += 1
        
        return self.state, {}
    
    def step(self, action):
        """
        Apply the action (an n-dimensional vector) and return:
          - observation: f(x)
          - reward: -|f(x_max) - f(x)| (f(x_max)=c)
          - done: always False (episodes only end by truncation)
          - truncated: True when max_episode_steps is reached
          - info: episode summary when truncated
        """
        self.tick += 1
        
        # Clip the action within the valid range.
        action = np.clip(action, self.x_range[0], self.x_range[1])
        self.x = action
        
        # Compute the current function value.
        y = self.compute_y(self.x)
        self.state = jnp.array([y], dtype=jnp.float32)
        
        # Reward based on how close f(x) is to the maximum value (c).
        reward = -abs(self.c - y)
        self.raw_rewards.append(reward)
        
        done = False
        truncated = self.tick >= self.max_episode_steps
        info = {}
        
        if truncated:
            info["final_observation"] = self.state
            info["episode_length"] = self.tick
            info["reward_per_episode"] = np.sum(self.raw_rewards)
            info["rewards"] = self.raw_rewards
            self.tick = 0
            self.raw_rewards = []
        
        return self.state, reward, done, truncated, info
    
    def compute_y(self, x):
        """
        Compute the polynomial value:
            f(x) = c - sum_{i=1}^{action_dim} (w_i * (x_i - x_max_i)**degree)
        """
        diff = x - self.x_max
        value = self.c - np.sum(self.weights * (diff ** self.degree))
        return value
    
    def render(self):
        """
        Print the current state and polynomial parameters.
        """
        print(f"Current x: {self.x}, f(x): {self.state[0]}, Peak at x_max: {self.x_max}, c: {self.c}, "
              f"weights: {self.weights}, degree: {self.degree}")
