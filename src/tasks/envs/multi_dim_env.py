import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp

class MultiDimEnv(gym.Env):
    """
    A Gymnasium environment where the agent interacts with a shifting quadratic function:
        f(x) = a * x^2 + b * x + c
    The peak (maximum) of the function varies across episodes.
    """
    def __init__(self, env_config=None, x_range=(-10, 10), max_episode_steps=12):
        super(MultiDimEnv, self).__init__()
        
        # Define range for x values.
        self.x_range = x_range
        self.max_episode_steps = max_episode_steps
        
        # Action space: agent selects a continuous x value.
        self.action_space = spaces.Box(
            low=np.array([self.x_range[0]]),
            high=np.array([self.x_range[1]]),
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space: the computed y value.
        self.observation_space = spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(1,), dtype=jnp.float32
        )
        
        self.tick = 0
        self.raw_rewards = []
        self.resetted = 0
        
        # Initialize state and action variable.
        self.state = None
        self.x = None
        
        # Initialize polynomial parameters.
        self.a = -1.0
        self.b = 0.0
        self.c = 10.0
        self.x_max = -self.b / (2 * self.a)  # Compute the peak position
        

    def shift_polynomial(self):
        """
        Randomize the polynomial parameters to create a new function.
        'a' should remain negative to ensure a mountain shape.
        """
        self.a = np.random.uniform(-2.0, -0.5)  # Keep a negative for a mountain shape
        self.b = np.random.uniform(-5.0, 5.0)   # Random slope
        self.c = np.random.uniform(5.0, 20.0)   # Random height
        
        # Compute the new peak position
        self.x_max = -self.b / (2 * self.a)

        # Ensure the peak is within the valid range
        self.x_max = np.clip(self.x_max, self.x_range[0], self.x_range[1])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Shift the polynomial parameters for a new episode.
        self.shift_polynomial()
        
        # Start with a random x value within the allowed range.
        self.x = np.random.uniform(self.x_range[0], self.x_range[1])
        y = self.compute_y(self.x)
        self.state = jnp.array([y], dtype=jnp.float32)
        y_max = self.compute_y(self.x_max)
        reward = -abs(y_max - y)
        comb = jnp.concatenate([self.state, jnp.array([self.x], dtype=jnp.float32), jnp.array([reward], dtype=jnp.float32)], axis=0)
        
        self.tick = 0
        self.raw_rewards = []
        self.resetted += 1
        
        # return comb, {}
        return self.state, {}

    def step(self, action):
        """
        Apply the action (choosing an x value) and return:
          - observation (y value)
          - reward (negative distance from the maximum at x_max)
          - done flag (always False, as episodes only truncate)
          - truncated flag (True when max_episode_steps is reached)
          - info (episode summary when truncated)
        """
        self.tick += 1
        
        # Ensure action is within the valid range.
        self.x = jnp.clip(action, self.x_range[0], self.x_range[1])
        
        # Compute the current y value.
        y = self.compute_y(self.x)
        self.state = jnp.array([y], dtype=jnp.float32)
        
        # Compute reward based on distance from the actual peak at x_max
        y_max = self.compute_y(self.x_max)
        reward = -abs(y_max - y)
        # print(f"reward: {reward}, y_max: {y_max}, y: {y}, x: {self.x}, x_max: {self.x_max}")
        self.raw_rewards.append(reward)
        
        done = False  # The task never ends naturally.
        truncated = self.tick >= self.max_episode_steps
        info = {}

        if truncated:
            info["final_observation"] = self.state
            info["episode_length"] = self.tick
            info["reward_per_episode"] = np.sum(self.raw_rewards)
            info["rewards"] = self.raw_rewards
            # Reset tick and rewards for the next episode.
            self.tick = 0
            self.raw_rewards = []
        
        # print("obs", self.state.shape, "rew", reward.shape, "act", action.shape)
        comb = jnp.concatenate([self.state, jnp.array([action], dtype=jnp.float32), jnp.array([reward], dtype=jnp.float32)], axis=0)
        # print(comb.shape)
        # print("c", comb, "\n", "state", self.state, "reward", reward, "action", action, "\n")
        
        # return comb, reward, done, truncated, info
        return self.state, reward, done, truncated, info


    def compute_y(self, x):
        """
        Compute the polynomial value at x: f(x) = a*x^2 + b*x + c.
        """
        return self.a * x**2 + self.b * x + self.c

    def render(self):
        """
        Print the current state and polynomial parameters.
        """
        print(f"Current x: {self.x}, f(x): {self.state[0]}, Peak at x_max: {self.x_max}, Parameters: a = {self.a}, b = {self.b}, c = {self.c}")
