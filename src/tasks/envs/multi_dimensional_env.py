import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
from gymnasium.spaces.utils import flatten, unflatten



class MultiDimEnv(gym.Env):
    """
    A Gymnasium environment where the agent interacts with a shifting quadratic function:
        f(x) = a * x^2 + b * x + c
    The peak (maximum) of the function varies across episodes.
    """
    def __init__(self, env_config=None, x_range=(-2, 2), max_episode_steps=1, batch_size=12, action_dim=2, degree=2):
        super(MultiDimEnv, self).__init__()
        
        # Define the allowed range for x values.
        self.x_range = x_range
        self.max_episode_steps = max_episode_steps
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.name = env_config['task']
        self.degree=degree
        
        # Action space: now a batch of continuous x values.
        self.action_space = spaces.Box(
                low=np.full((self.batch_size, self.action_dim), self.x_range[0]),
                high=np.full((self.batch_size, self.action_dim), self.x_range[1]),
                dtype=np.float32
            )
        
        # self.action_space = spaces.Dict() # this is what I want to expand on spaces box of continuous action values values, observation is a continious scalar value values and the reward is a scalar value
        
        
        # Observation space: a batch of computed y values.
        self.observation_space = spaces.Dict({
            "actions": spaces.Box(
                low=np.full((self.batch_size, self.action_dim), self.x_range[0]),
                high=np.full((self.batch_size, self.action_dim), self.x_range[1]),
                dtype=np.float32
            ),
            "observations": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.batch_size,1), 
                dtype=np.float32
            ),
            "reward": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(1,), 
                dtype=np.float32
            )
        })
        
        
        self.tick = 0
        self.raw_rewards = []
        self.scaled_rewards = []
        self.best_rewards = []
        self.resetted = 0
        
        # Initialize state and x.
        self.state = None
        
        # Initialize polynomial parameters.
        self.x_max = None
        self.c = None
        self.weights = None
        self.shift_polynomial()

    
    def shift_polynomial(self):
        # Randomize the maximum location uniformly for each dimension.
        self.x_max = np.random.uniform(self.x_range[0], self.x_range[1], size=(1,self.action_dim))
        # Randomize the constant such that f(x_max) = c, and weights.
        self.c = np.random.uniform(5.0, 20.0)
        self.weights = np.random.uniform(0.5, 2.0, size=(self.action_dim,))
        # self.x_max = np.random.uniform(0.0, 0.0, size=(1,self.action_dim))
        # # Randomize the constant such that f(x_max) = c, and weights.
        # self.c = np.random.uniform(10.0, 10.0)
        # self.weights = np.random.uniform(3.0, 3.0, size=(self.action_dim,))
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Optionally shift polynomial parameters if desired.
        self.shift_polynomial()
        
        # Sample a batch of random x values within the allowed range.
        action = np.random.uniform(self.x_range[0], self.x_range[1], size=(self.batch_size, self.action_dim))
        # action = jnp.zeros((self.batch_size, self.action_dim))
        y = self.compute_y(action)
        self.max_y = self.compute_y(self.x_max)
        # print("resetting the params", self.x.shape, y.shape, "\n")
        self.state = jnp.array(y, dtype=jnp.float32)
        # self.state = jnp.squeeze(self.state)
        self.state = jnp.expand_dims(self.state, axis=-1)
        self.state = self.state.reshape(self.batch_size, 1)
        
        
        reward = jnp.sum(-jnp.abs(self.max_y - y))#, axis=0)  # elementwise operation over the batch
        s_reward = reward / self.batch_size
        reward = jnp.expand_dims(reward, axis=-1)
        # comb = np.concatenate([self.x, self.state], axis=-1)
        
        self.tick = 0
        self.raw_rewards = []
        self.scaled_rewards = []
        self.best_rewards = [s_reward]
        self.resetted += 1
        
        # print(self.x.shape, self.state.shape, reward.shape)
        
        
        observation = {
            "actions": np.array(action, dtype=np.float32),
            "observations": np.array(self.state, dtype=np.float32),
            "reward": np.array(reward, dtype=np.float32)
        }
        # print("Observation types:", {k: type(v) for k, v in observation.items()})
        # observation = np.ones((3,2))
        return observation, {}

    def step(self, action):
        """
        Apply the batch of actions (each is an x value) and return:
          - observation (batch of y values)
          - reward (batch of negative distances from the maximum at x_max)
          - done flag (always False, as episodes only truncate)
          - truncated flag (True when max_episode_steps is reached)
          - info (episode summary when truncated)
        """
        # print("action", action)
        self.tick += 1
        # print("action", action, "x", self.x, "state", self.state, "\n")
        # print("actuibs", action.shape)
        # print("actions", action.shape)
        # action = np.random.uniform(self.x_range[0], self.x_range[1], size=(self.batch_size, self.action_dim))
        
        action = jnp.reshape(action, (self.batch_size, self.action_dim))
        # assert action.shape == (self.batch_size, 1), f"Expected shape {(self.batch_size, 1)}, got {action.shape}"
        # Ensure action is a JAX array and clip each element to be within x_range.
        action = jnp.array(action)
        action = jnp.clip(action, self.x_range[0], self.x_range[1])
        # Compute the batch of y values.
        y = self.compute_y(action)
        # print("stepping along", self.x.shape, y.shape, action.shape,"\n")
        self.state = jnp.array(y, dtype=jnp.float32)
        self.state = jnp.expand_dims(self.state, axis=-1)
        self.state = self.state.reshape(self.batch_size, 1)
        
        
        # Compute reward for each sample:
        # The peak value is computed from self.x_max (a scalar) so the difference is broadcast.
        reward = jnp.sum(-jnp.abs(self.max_y - y))  # elementwise operation over the batch
        e_reward = jnp.expand_dims(reward, axis=0)
        s_reward = reward / self.batch_size
        
        self.raw_rewards.append(reward)
        self.scaled_rewards.append(s_reward)
        if self.best_rewards[0] < s_reward:
            self.best_rewards[0] = s_reward
        # print("reward", reward.shape, self.max_y.shape, y.shape, (-jnp.abs(self.max_y - y)).shape, e_reward.shape)

        
        # print("action", action, "reward", reward, "y", y, "y_max", self.max_y, "x_max", self.x_max, "\n")

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
            info["s_rewards"] = self.scaled_rewards
            info["best_rewards"] = self.best_rewards
            # Reset tick and rewards for the next episode.
            self.tick = 0
            self.raw_rewards = []
            self.scaled_rewards = []
            self.best_rewards = []
            
        # reward = jnp.expand_dims(reward, axis=-1)
            

        # comb = np.concatenate([self.x, self.state], axis=-1)
        observation = {
            "actions": np.array(action, dtype=np.float32),
            "observations": np.array(self.state, dtype=np.float32),
            "reward": np.array(e_reward, dtype=np.float32)
        }
      
            
        return observation, reward, done, truncated, info

    def compute_y(self, x):
        """
        Compute the polynomial value:
            f(x) = c - sum_{i=1}^{action_dim} (w_i * (x_i - x_max_i)**degree)
        """
        
        diff = x - self.x_max  # Compute difference
        weighted_term = self.weights * (diff ** self.degree)  # Apply weights
        result = self.c - np.sum(weighted_term, axis=1)  # Sum across dimensions
        return result.squeeze()  # Convert (1,) to scalar if needed

    def render(self):
        """
        Print the current state and polynomial parameters.
        """
        print(f"Current x: {self.x}\n"
              f"f(x): {self.state}\n"
              f"Peak at x_max: {self.x_max}\n"
              f"Parameters: a = {self.a}, b = {self.b}, c = {self.c}")
