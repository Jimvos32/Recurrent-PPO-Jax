import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
from gymnasium.spaces.utils import flatten, unflatten
import jax



class MultiMask(gym.Env):
    """
    A Gymnasium environment where the agent interacts with a shifting quadratic function:
        f(x) = a * x^2 + b * x + c
    The peak (maximum) of the function varies across episodes.
    """
    def __init__(self, env_config=None):
        super(MultiMask, self).__init__()
        # Define the allowed range for x values.
        self.x_range = env_config["bounds"]
        self.max_episode_steps = env_config["max_episode_steps"]
        self.batches = env_config["batches"]
        self.max_batches = np.max(self.batches)
        self.batch_size = self.max_batches
        self.action_dim = env_config['action_dim']
        self.name = env_config['task']
        self.degree=env_config['degree']
        
        # Action space: now a batch of continuous x values.
        self.action_space = spaces.Box(
                low=np.full((self.max_batches, self.action_dim), self.x_range[0]),
                high=np.full((self.max_batches, self.action_dim), self.x_range[1]),
                dtype=np.float32
            )
        
        # self.action_space = spaces.Dict() # this is what I want to expand on spaces box of continuous action values values, observation is a continious scalar value values and the reward is a scalar value
        
        
        # Observation space: a batch of computed y values.
        self.observation_space = spaces.Dict({
            "actions": spaces.Box(
                low=np.full((self.max_batches, self.action_dim), self.x_range[0]),
                high=np.full((self.max_batches, self.action_dim), self.x_range[1]),
                dtype=np.float32
            ),
            "observations": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.max_batches,1), 
                dtype=np.float32
            ),
            "reward": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(1,), 
                dtype=np.float32
            ),
            "mask": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(1,), 
                dtype=np.int32
            )
        })
        
        
        self.tick = 0
        self.raw_rewards = []
        self.scaled_rewards = []
        self.best_rewards = []
        self.scaled_diff = []
        self.scaled_obs = []
        self.mse = []
        self.resetted = 0
        
        # Initialize state and x.
        self.state = None
        self.x = None
        
        # Initialize polynomial parameters.
        self.x_max = None
        self.y_min = None
        self.max_y = None
        
        self.c = None
        self.weights = None
        self.mask = None
        self.shift_polynomial()

    
    def shift_polynomial(self):
        # Randomize the maximum location uniformly for each dimension.
        self.x_max = np.random.uniform(self.x_range[0], self.x_range[1], size=(1,self.action_dim))
        # Randomize the constant such that f(x_max) = c, and weights.
        self.c = np.random.uniform(5.0, 20.0)
        self.weights = np.random.uniform(0.5, 2.0, size=(self.action_dim,))
        
     
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Optionally shift polynomial parameters if desired.
        # self.shift_polynomial()
        self.batch_size = np.random.choice(self.batches)
        
       
        # Sample a batch of random x values within the allowed range.
        self.x = np.random.uniform(self.x_range[0], self.x_range[1], size=(self.max_batches, self.action_dim))
        # print(self.batch_size, "-1")
        y = self.compute_y(self.x)
        # print(self.batch_size, "0")
        self.y_min = self.find_minimum()
        self.max_y = self.compute_y(self.x_max)
        # print("resetting the params", self.x.shape, y.shape, "\n")
        self.state = jnp.array(y, dtype=jnp.float32)
        # self.state = jnp.squeeze(self.state)
        self.state = jnp.expand_dims(self.state, axis=-1)
        self.state = self.state.reshape(self.max_batches, 1)
        
        self.x = jnp.array(self.x, dtype=jnp.float32)
        self.state = jnp.array(self.state, dtype=jnp.float32)
        # self.x = self.x.at[self.batch_size:, :].set(jnp.zeros_like(self.x[self.batch_size:, :]))
        # self.state = self.state.at[self.batch_size:, :].set(jnp.zeros_like(self.state[self.batch_size:, :]))
        # scaled_difference = jnp.abs(self.max_y - self.state)[:self.batch_size, :] / self.batch_size 
        
        
        # print("action", self.max_y , "-  ", self.state, " /", self.batch_size," = ", scaled_difference)
        # print(y.shape, self.max_y.shape, "test")
        # print((-jnp.abs(self.max_y - y)).shape, "fas")
        reward = jnp.sum(-jnp.abs(self.max_y - y))#, axis=0)  # elementwise operation over the batch
        # print(self.batch_size, "2")
        reward = jnp.expand_dims(reward, axis=-1)
        # comb = np.concatenate([self.x, self.state], axis=-1)
        
        self.tick = 0
        self.raw_rewards = []
        self.best_rewards = self.state
        self.resetted += 1
        
        # print(self.x.shape, self.state.shape, reward.shape)
        # print(self.batch_size.shape)
        # print("joj", jnp.expand_dims(self.batch_size, axis=-1).shape)
        
        observation = {
            "actions": np.array(self.x, dtype=np.float32),
            "observations": np.array(self.state, dtype=np.float32),
            "reward": np.array(reward, dtype=np.float32),
            "mask": np.array(jnp.expand_dims(self.batch_size, axis=-1), dtype=np.int32)
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
        self.tick += 1
        
        action = jnp.reshape(action, (self.max_batches, self.action_dim))
      
        action = jnp.array(action)
        self.x = jnp.clip(action, self.x_range[0], self.x_range[1])
        # Compute the batch of y values.
        y = self.compute_y(self.x)
        # print("stepping along", self.x.shape, y.shape, action.shape,"\n")
        self.state = jnp.array(y, dtype=jnp.float32)
        self.state = jnp.expand_dims(self.state, axis=-1)
        self.state = self.state.reshape(self.max_batches, 1)
        
        difference = self.max_y - self.state[:self.batch_size, :] 
        squared_difference = jnp.square(difference)
        
        mse = jnp.mean(squared_difference, axis=0)
        
        
        # print("batch", self.batch_size, "scaled_difference", scaled_difference, "max", self.max_y, "action", self.state)
        # print(self.state.shape, self.state[:self.batch_size, :].shape)
        # jax.debug.print("max y {} - {} = {}", self.max_y, self.state, difference)
       
        e_reward = mse
        reward = jnp.squeeze(e_reward, axis=-1)
        
        # regret = jnp.abs(self.max_y - self.state)
        c = self.state[:self.batch_size, :]
        
        scaled_observation = (self.state[:self.batch_size, :] - self.y_min) / (self.max_y - self.y_min)
        scaled_difference = (difference - self.y_min) / (self.max_y - self.y_min)
        
        # print("scaled_observation", scaled_observation.shape, "scaled_difference", scaled_difference.shape, "diff", difference.shape, c.shape)
        
        avg_scl_obs = jnp.mean(scaled_observation, axis=0)
        avg_scl_diff = jnp.mean(scaled_difference, axis=0)
        
        
        # print("avg_scl_obs", avg_scl_obs.shape, "avg_scl_diff", avg_scl_diff.shape)
        
        
        max_sample = jnp.max(self.state, axis=0)
        # jax.debug.print("max_sample {} {}", max_sample, self.state)
        
        if self.best_rewards[0] < max_sample:
            self.best_rewards = self.best_rewards.at[0].set(max_sample)
        
      
        
        
        self.raw_rewards.append(reward)
        self.scaled_obs.append(avg_scl_obs)
        self.scaled_diff.append(avg_scl_diff)
        self.mse.append(mse)
      

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
            info["batch_mse"] = jnp.mean(jnp.array(self.mse), axis=0)
            info["last_scaled_diff"] = avg_scl_diff
            info["last_scaled_obs"] = avg_scl_obs
            info["scaled_diff"] = jnp.mean(jnp.array(self.scaled_diff), axis=0)
            info["scaled_obs"] = jnp.mean(jnp.array(self.scaled_obs), axis=0)
            info["best_rewards"] = self.best_rewards
            info["success"] = ((self.best_rewards[0] - self.y_min) / (self.max_y - self.y_min)) > 0.9
            # Reset tick and rewards for the next episode.
            self.tick = 0
            self.raw_rewards = []
            
        # reward = jnp.expand_dims(reward, axis=-1)
            

        # comb = np.concatenate([self.x, self.state], axis=-1)
        observation = {
            "actions": np.array(self.x, dtype=np.float32),
            "observations": np.array(self.state, dtype=np.float32),
            "reward": np.array(e_reward, dtype=np.float32),
            "mask": np.array(jnp.expand_dims(self.batch_size, axis=-1), dtype=np.int32)

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
    
    def find_minimum(self):
        # Unpack the lower and upper bounds (assumed to be scalars)
        lower, upper = self.x_range
        
        # For each dimension, choose the bound that is farther from the maximum
        # Note: self.x_max has shape (1, action_dim) and broadcasting is used here.
        x_min = np.where((self.x_max - lower) > (upper - self.x_max), lower, upper)
        
        # Compute the function value at this minimum
        y_min = self.compute_y(x_min)
        return y_min

    def render(self):
        """
        Print the current state and polynomial parameters.
        """
        print(f"Current x: {self.x}\n"
              f"f(x): {self.state}\n"
              f"Peak at x_max: {self.x_max}\n"
              f"Parameters: a = {self.a}, b = {self.b}, c = {self.c}")
