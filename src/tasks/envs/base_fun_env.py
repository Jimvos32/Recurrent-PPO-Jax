import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
import jax
import abc

class BaseOptimizationEnv(gym.Env, abc.ABC):
    """
    Base environment for optimization tasks.
    This abstract class provides common functionality for environments where
    the agent interacts with a function, including resetting, stepping, and
    mapping actions to bounds. Subclasses must implement:
      - initialize_function(): to set function-specific parameters (e.g. optimum point, min value)
      - compute_y(x): to compute the function value for input x.
    """
    def __init__(self, env_config=None):
        super().__init__()
        # Common environment configuration
        self.x_range = env_config["bounds"]
        self.total_samples = env_config["total_episode_samples"]
        self.max_episode_steps = env_config["max_episode_steps"]
        self.batches = env_config["batches"]
        self.max_batches = env_config["max_batches"]
        self.batch_size = self.max_batches
        self.action_dim = env_config["action_dim"]
        self.name = env_config["task"]
        self.random = env_config["random"]  
        # Adjust max_episode_steps based on total samples and batch size.
        self.max_episode_steps = self.total_samples // self.batch_size

        # Reward scaling parameters (common across environments)
        self.r_best = 0.1
        self.r_impr = 0.2
        self.r_avg = 0.0
        self.r_new_best = 0.2
        self.r_mse = 0.0
        self.r_obs = 0.0
        self.b_pen = 0.0
        self.r_suc = 12.0
        self.r_scale = 10

        # Define action and observation spaces.
        self.action_space = spaces.Box(
            low=np.full((self.max_batches, self.action_dim), self.x_range[0]),
            high=np.full((self.max_batches, self.action_dim), self.x_range[1]),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "actions": spaces.Box(
                low=np.full((self.max_batches, self.action_dim), self.x_range[0]),
                high=np.full((self.max_batches, self.action_dim), self.x_range[1]),
                dtype=np.float32
            ),
            "observations": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_batches, 1),
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
            ),
            "step": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(1,),
                dtype=np.int32
            )
        })
        self.tick = 0
        # self.initialize_function()

    @abc.abstractmethod
    def initialize_function(self):
        """
        Initialize function-specific parameters.
        This method must set attributes such as:
          - self.optimum_point: the input at which the function attains its maximum.
          - self.min_y: the minimum value of the function (used for scaling).
        """
        pass

    @abc.abstractmethod
    def compute_y(self, x):
        """
        Compute the function value given input x.
        """
        pass

    def map_to_bounds(self, actions):
        shaped_action = jnp.reshape(actions, (self.max_batches, self.action_dim))
        lower, upper = self.x_range[0], self.x_range[1]
        scale = (upper - lower) / 2
        shift = (upper + lower) / 2
        return scale * shaped_action + shift

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.initialize_function()  # Set function parameters (must define optimum_point and min_y)
        self.batch_size = np.random.choice(self.batches)
        # Initial action: a batch of zeros.
        if self.random is None:
            action = np.random.uniform(self.x_range[0], self.x_range[1], size=(self.max_batches, self.action_dim))
        else:
            action = np.zeros((self.max_batches, self.action_dim), dtype=np.float32)
        obs = self.compute_y(action)
        obs = jnp.array(obs, dtype=jnp.float32).reshape(self.max_batches, 1)
        # Compute the optimum function value at self.optimum_point.
        
        self.max_y = self.compute_y(self.optimum_point)
        # Scale the observation between the min and max values.
        scaled_observation = jnp.clip((obs[:self.batch_size, :] - self.min_y) / (self.max_y - self.min_y),a_min=0, a_max=1)
        max_sample = jnp.max(obs, axis=0)
        scaled_max = jnp.clip((max_sample - self.min_y) / (self.max_y - self.min_y), a_min=0, a_max=1)
        avg_obs = jnp.mean(scaled_observation, axis=0)
        self.scaled_obs = [avg_obs]
        new_best = scaled_max
        reward = (self.r_best * scaled_max + self.r_impr * avg_obs + new_best * self.r_new_best) * self.r_scale
        self.tick = 0
        self.raw_rewards = []
        self.best_rewards = scaled_max
        self.scaled_diff = []
        self.mse = []
        self.actions = []
        self.eval_obs = []
        observation = {
            "actions": np.array(action, dtype=np.float32),
            "observations": np.array(obs, dtype=np.float32),
            "reward": np.array(reward, dtype=np.float32),
            "mask": np.array(jnp.expand_dims(self.batch_size, axis=-1), dtype=np.int32),
            "step": np.zeros((1,), dtype=np.int32)
        }
        return observation, {}

    def step(self, action):
        self.tick += 1
        
        action = jnp.reshape(action, (self.max_batches, self.action_dim))
        assert action.shape == (self.max_batches, self.action_dim), \
            f"Expected shape {(self.max_batches, self.action_dim)}, got {action.shape}"
        action = self.map_to_bounds(action)
        if self.random:
            action = np.random.uniform(self.x_range[0], self.x_range[1], size=(self.max_batches, self.action_dim))
        
        self.actions.append(action)
        
        obs = self.compute_y(action)
        obs = jnp.array(obs, dtype=jnp.float32).reshape(self.max_batches, 1)
        
        difference = self.max_y - obs[:self.batch_size, :]
        mse = jnp.mean(jnp.square(difference), axis=0)
        
        scaled_observation = jnp.clip((obs[:self.batch_size, :] - self.min_y) / (self.max_y - self.min_y), a_min=0, a_max=1)
        scaled_difference = jnp.clip(difference / (self.max_y - self.min_y), a_min=0, a_max=1)
        avg_scl_obs = jnp.mean(scaled_observation, axis=0)
        avg_scl_diff = jnp.mean(scaled_difference, axis=0)
        
        max_sample = jnp.max(obs, axis=0)
        scaled_max = jnp.clip((max_sample - self.min_y) / (self.max_y - self.min_y), a_min=0, a_max=1)
        
        avg_imp = jnp.mean(scaled_observation - self.scaled_obs[-1], axis=0)
        new_best = jnp.maximum(0.0, max_sample - self.best_rewards[0])
        
        suc_reward = 0.0
        if self.best_rewards > 0.95 and self.tick >= self.max_episode_steps:
            suc_reward = self.r_suc

        s_new_best = new_best / (self.max_y - self.min_y)
        e_reward = (self.r_best * scaled_max + self.r_impr * avg_imp +
                    s_new_best * self.r_new_best - self.r_mse * mse +
                    self.r_obs * avg_scl_obs + suc_reward) * self.r_scale
        reward = jnp.squeeze(e_reward, axis=-1)
        if self.best_rewards < scaled_max:
            self.best_rewards = scaled_max
        self.raw_rewards.append(reward)
        self.scaled_obs.append(avg_scl_obs)
        self.scaled_diff.append(avg_scl_diff)
        self.eval_obs.append(scaled_observation)
        self.mse.append(mse)
        done = False
        truncated = self.tick >= self.max_episode_steps
        info = {}
        if truncated:
            info["final_observation"] = obs
            info["episode_length"] = self.tick
            # print("rae", jnp.array(self.raw_rewards))
            total_reward = jnp.sum(jnp.array(self.raw_rewards), axis=0)
            info["reward_per_episode"] = total_reward
            info["rewards"] = self.raw_rewards
            info["batch_mse"] = jnp.mean(jnp.array(self.mse), axis=0)
            info["last_scaled_diff"] = avg_scl_diff
            info["last_scaled_obs"] = avg_scl_obs
            info["scaled_diff"] = jnp.mean(jnp.array(self.scaled_diff), axis=0)
            info["scaled_obs"] = jnp.mean(jnp.array(self.scaled_obs[2:]), axis=0)
            info["best_rewards"] = self.best_rewards
            info["actions"] = self.actions
            info["eval_scaled_diff"] = jnp.array(self.eval_obs)
            info["success"] = self.best_rewards > 0.95
            info["max_x"] = self.optimum_point
            # Reset counters and lists for the next episode.
            self.tick = 0
            self.raw_rewards = []
            self.scaled_diff = []
            self.scaled_obs = []
            self.mse = []
            self.actions = []
            self.eval_obs = []
        observation = {
            "actions": np.array(action, dtype=np.float32),
            "observations": np.array(obs, dtype=np.float32),
            "reward": np.array(e_reward, dtype=np.float32),
            "mask": np.array(jnp.expand_dims(self.batch_size, axis=-1), dtype=np.int32),
            "step": np.array([self.tick], dtype=np.int32)
        }
        return observation, reward, done, truncated, info

    def render(self):
        print(f"Current actions: {self.actions}\n")




