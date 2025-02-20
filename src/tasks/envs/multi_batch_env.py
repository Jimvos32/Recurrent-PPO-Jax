import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp

class MultiSampleEnv(gym.Env):
    """
    A Gymnasium environment where the agent interacts with a shifting quadratic function:
        f(x) = a * x^2 + b * x + c
    The peak (maximum) of the function varies across episodes.
    """
    def __init__(self, env_config=None, x_range=(-10, 10), max_episode_steps=12, batch_size=1):
        super(MultiSampleEnv, self).__init__()
        
        # Define the allowed range for x values.
        self.x_range = x_range
        self.max_episode_steps = max_episode_steps
        self.batch_size = batch_size
        
        # Action space: now a batch of continuous x values.
        self.action_space = spaces.Box(
            low=np.array([[self.x_range[0]]] * self.batch_size),
            high=np.array([[self.x_range[1]]] * self.batch_size),
            shape = (self.batch_size, 1),
            dtype=np.float32
        )
        
        
        # Observation space: a batch of computed y values.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.batch_size, 3), dtype=np.float32
        )
        
        self.tick = 0
        self.raw_rewards = []
        self.resetted = 0
        
        # Initialize state and x.
        self.state = None
        self.x = None
        
        # Initialize polynomial parameters.
        self.a = -1.0
        self.b = 2.0
        self.c = 10.0
        self.x_max = -self.b / (2 * self.a)  # Compute the peak position
        self.max_y = self.compute_y(self.x_max)  # Compute the peak value

    
    def shift_polynomial(self):
        self.a = np.random.uniform(-2.0, -0.5)  # Keep a negative for a mountain shape
        self.b = np.random.uniform(-5.0, 5.0)   # Random slope
        self.c = np.random.uniform(5.0, 20.0)   # Random height
        
        # Compute the new peak position
        self.x_max = -self.b / (2 * self.a)

        # Ensure the peak is within the valid range
        self.x_max = np.clip(self.x_max, self.x_range[0], self.x_range[1])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Optionally shift polynomial parameters if desired.
        self.shift_polynomial()
        
        # Sample a batch of random x values within the allowed range.
        self.x = np.random.uniform(self.x_range[0], self.x_range[1], size=(self.batch_size, 1))
        y = self.compute_y(self.x)
        self.max_y = self.compute_y(self.x_max)
        # print("resetting the params", self.x.shape, y.shape, "\n")
        self.state = jnp.array(y, dtype=jnp.float32)
        
        reward = jnp.sum(-jnp.abs(self.max_y - y), axis=0)  # elementwise operation over the batch
        reward = jnp.expand_dims(reward, axis=-1)
            
        comb = np.concatenate([self.x, self.state, reward], axis=-1)
        
        
        self.tick = 0
        self.raw_rewards = []
        self.resetted += 1
        
        
        
        return comb, {}

    def step(self, action):
        """
        Apply the batch of actions (each is an x value) and return:
          - observation (batch of y values)
          - reward (batch of negative distances from the maximum at x_max)
          - done flag (always False, as episodes only truncate)
          - truncated flag (True when max_episode_steps is reached)
          - info (episode summary when truncated)
        """
        self.tick += 1
        # print("action", action, "x", self.x, "state", self.state, "\n")
        action = jnp.reshape(action, (self.batch_size, 1))
        assert action.shape == (self.batch_size, 1), f"Expected shape {(self.batch_size, 1)}, got {action.shape}"
        # Ensure action is a JAX array and clip each element to be within x_range.
        action = jnp.array(action)
        self.x = jnp.clip(action, self.x_range[0], self.x_range[1])
        
        # Compute the batch of y values.
        y = self.compute_y(self.x)
        # print("stepping along", self.x.shape, y.shape, action.shape,"\n")
        self.state = jnp.array(y, dtype=jnp.float32)
        
        # Compute reward for each sample:
        # The peak value is computed from self.x_max (a scalar) so the difference is broadcast.
        reward = jnp.sum(-jnp.abs(self.max_y - y), axis=0)  # elementwise operation over the batch
        s_reward = jnp.squeeze(reward, axis=-1)
        self.raw_rewards.append(reward)
        
        # print("action", action, "reward", reward, "y", y, "y_max", self.max_y, "\n")
        
        done = False  # Episodes do not naturally end; only truncated.
        truncated = self.tick >= self.max_episode_steps
        info = {}

        if truncated:
            info["final_observation"] = self.state
            info["episode_length"] = self.tick
            # Sum rewards across time steps for each batch element.
            total_reward = jnp.sum(jnp.array(self.raw_rewards), axis=0)
            info["reward_per_episode"] = total_reward
            info["rewards"] = self.raw_rewards
            # Reset tick and rewards for the next episode.
            self.tick = 0
            self.raw_rewards = []
            
        reward = jnp.expand_dims(reward, axis=-1)
            
       
        comb = np.concatenate([self.x, self.state, reward], axis=-1)
        
        # print("this is the work", s_reward.shape, comb.shape, self.state.shape)
            
        return comb, s_reward, done, truncated, info

    def compute_y(self, x):
        """
        Compute the polynomial value at x: f(x) = a*x^2 + b*x + c.
        This function works with both scalar and batch (array) inputs.
        """
        return self.a * x**2 + self.b * x + self.c

    def render(self):
        """
        Print the current state and polynomial parameters.
        """
        print(f"Current x: {self.x}\n"
              f"f(x): {self.state}\n"
              f"Peak at x_max: {self.x_max}\n"
              f"Parameters: a = {self.a}, b = {self.b}, c = {self.c}")
