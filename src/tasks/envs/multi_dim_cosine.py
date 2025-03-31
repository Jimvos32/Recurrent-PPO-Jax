import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
from gymnasium.spaces.utils import flatten, unflatten
import jax



class MultiCosine(gym.Env):
    """
    A Gymnasium environment where the agent interacts with a shifting quadratic function:
        f(x) = a * x^2 + b * x + c
    The peak (maximum) of the function varies across episodes.
    """
    def __init__(self, env_config=None):
        super(MultiCosine, self).__init__()
        # Define the allowed range for x values.
        self.x_range = env_config["bounds"]
        self.max_episode_steps = env_config["max_episode_steps"]
        self.batches = env_config["batches"]
        self.max_batches = env_config["max_batches"]
        self.batch_size = self.max_batches
        self.action_dim = env_config['action_dim']
        self.name = env_config['task']
        self.degree=env_config['degree']
        self.max=env_config['max_episode_steps']
        self.num_oscillations = env_config['num_oscillations']
       
        # print("batch", self.batches)
        
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
            ),
            "step": spaces.Box(
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
        self.actions = []
        self.eval_obs = []
        self.resetted = 0
        
        self.r_best = 0.7
        self.r_impr = 0.2
        self.r_avg = 0.1
        self.r_new_best = 0.5
        self.r_mse = 0.0
        self.r_obs = 0.0
        
        # self.r_best = 0.0
        # self.r_impr = 0.0
        # self.r_avg = 0.0
        # self.r_new_best = 0.0
        # self.r_mse = 1.0
        # self.r_obs = 0.0
        
        
       
        
        # Initialize polynomial parameters.
        self.max_x = None
        self.min_y = None
        self.max_y = None
        
        self.c = None
        self.weights = None
        self.mask = None
        self.randomise_function()

    
    def randomise_function(self):
        # self.c = np.random.uniform(5.0, 20.0)  # Offset
        # self.A0 = np.random.uniform(10.0, 20.0)  # Large peak amplitude
        # self.B0 = np.random.uniform(0.5, 1.5, size=(self.action_dim,))  # Large peak frequency
        # self.s0 = np.random.uniform(self.x_range[0], self.x_range[1], size=(self.action_dim,))  # Peak location

        # # Small oscillations (randomized)
        # self.small_A = np.random.uniform(0.2, 3.0, size=(self.action_dim, self.num_oscillations))
        # self.small_B = np.random.uniform(0.5, 4.0, size=(self.action_dim, self.num_oscillations))
        
        self.c = np.random.uniform(10.0, 10.0)  # Offset
        self.A0 = np.random.uniform(15.0, 15.0)  # Large peak amplitude
        self.B0 = np.random.uniform(1.1, 1.1, size=(self.action_dim,))  # Large peak frequency
        self.s0 = np.random.uniform(0.5, 0.5, size=(self.action_dim,))  # Peak location

        # Small oscillations (randomized)
        self.small_A = np.random.uniform(2.0, 2.0, size=(self.action_dim, self.num_oscillations))
        self.small_B = np.random.uniform(3.0, 3.0, size=(self.action_dim, self.num_oscillations))
        
        
        # Force all oscillations to peak at s0 (ensuring known max)
        self.small_shift = np.tile(self.s0.reshape(-1, 1), (1, self.num_oscillations))  # Align peaks
        self.small_phase = np.zeros((self.action_dim, self.num_oscillations))  # No phase shifts

        # Compute known global max value (since all cosines are +1 at s0)
        self.max_x = self.s0
        self.max_y = self.c + self.A0 * self.action_dim + np.sum(self.small_A)

        # Compute theoretical min bound (all cosines at -1)
        self.min_y = self.c - self.A0 * self.action_dim - np.sum(self.small_A)

        # Estimate empirical min by grid search
        # self.estimate_empirical_min()
        
        
    
        
     
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Optionally shift polynomial parameters if desired.
        self.randomise_function()
        self.batch_size = np.random.choice(self.batches)
        
       
        # Sample a batch of random x values within the allowed range.
        # action = np.random.uniform(self.x_range[0], self.x_range[1], size=(self.max_batches, self.action_dim))
        action = np.zeros((self.batch_size, self.action_dim))
        # print(self.batch_size, "-1")
        obs = self.compute_y(action)
        # print(self.batch_size, "0")
        # self.y_min = self.find_minimum()
        self.max_y = self.compute_y(self.max_x)
        # print("resetting the params", self.x.shape, y.shape, "\n")
        obs = jnp.array(obs, dtype=jnp.float32)
        # self.state = jnp.squeeze(self.state)
        obs = jnp.expand_dims(obs, axis=-1)
        obs = obs.reshape(self.max_batches, 1)
        
        action = jnp.array(action, dtype=jnp.float32)
        scaled_observation = jnp.clip((obs[:self.batch_size, :] - self.min_y) / (self.max_y - self.min_y), a_min=0, a_max=1)
        
        max_sample = jnp.max(obs, axis=0)
        scaled_max  = jnp.clip((max_sample - self.min_y) / (self.max_y - self.min_y), a_min=0, a_max=1)
        
        avg_obs = jnp.mean(scaled_observation)
        self.scaled_obs.append(avg_obs)
        
        # print(type(scaled_max), "1", scaled_max.shape)
        # bb = scaled_max[0]
        # new_best = jnp.max(jnp.array([0.0, bb]))
        new_best = jnp.maximum(0.0, scaled_max)
        # print("new_best", new_best)
        
        reward = self.r_best * max_sample + self.r_impr * avg_obs + new_best * self.r_new_best
        # reward = jnp.sum(-jnp.abs(self.max_y - obs))#, axis=0)  # elementwise operation over the batch
        # e_reward = jnp.squeeze(reward, axis=-1)
        # comb = np.concatenate([self.x, self.state], axis=-1)
        # print("reward", reward.shape, "e_reward", e_reward.shape)
        
        self.tick = 0
        self.raw_rewards = []
        self.best_rewards = obs
        self.resetted += 1
        
        # print(self.x.shape, self.state.shape, reward.shape)
        # print(self.batch_size.shape)
        # print("joj", jnp.expand_dims(self.batch_size, axis=-1).shape)
        
        observation = {
            "actions": np.array(action, dtype=np.float32),
            "observations": np.array(obs, dtype=np.float32),
            "reward": np.array(reward, dtype=np.float32),
            "mask": np.array(jnp.expand_dims(self.batch_size, axis=-1), dtype=np.int32),
            "step": np.zeros((1,), dtype=np.int32)
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
        copy = action
        # assert action.shape == (self.batch_size, self.action_dim), f"Expected shape {(self.batch_size, self.action_dim)}, got {action.shape}"
        action = self.map_to_bounds(action)
        # action = np.random.uniform(self.x_range[0], self.x_range[1], size=(self.max_batches, self.action_dim))
        self.actions.append(action)
        # action = jnp.reshape(action, (self.max_batches, self.action_dim))
      
        # action = jnp.array(action)
        # self.x = jnp.clip(action, self.x_range[0], self.x_range[1])
        # Compute the batch of y values.
        obs = self.compute_y(action)
        # print("stepping along", self.x.shape, y.shape, action.shape,"\n")
        obs = jnp.array(obs, dtype=jnp.float32)
        obs = jnp.expand_dims(obs, axis=-1)
        obs = obs.reshape(self.max_batches, 1)
        
        
        
        difference = self.max_y - obs[:self.batch_size, :] 
        squared_difference = jnp.square(difference)
        
        mse = jnp.mean(squared_difference, axis=0)
        
        
        
        # print(self.state.shape, self.state[:self.batch_size, :].shape)
        # jax.debug.print("max y {} - {} = {}", self.max_y, self.state, difference)
       
        # e_reward = mse * -1
        # reward = jnp.squeeze(e_reward, axis=-1)
        # print("tge e_reward", e_reward.shape, "reward the", reward.shape)
        
        # jax.debug.print("reward {}\ny {}\nact {}\nout {}\nmax y {}\nmax_x {}\n", reward, y, action, copy, self.max_y, self.x_max)
        
        # regret = jnp.abs(self.max_y - self.state)
        # c = self.state[:self.batch_size, :]
        
        scaled_observation = jnp.clip((obs[:self.batch_size, :] - self.min_y) / (self.max_y - self.min_y), a_min=0, a_max=1)
        scaled_difference = jnp.clip(difference / (self.max_y - self.min_y), a_min=0, a_max=1)
        # jax.debug.print("my {} miny {}, obs {} diff{} scaled fi {}",self.max_y, self.y_min, self.state[:self.batch_size, :], difference,scaled_difference)
        # print("scaled_observation", scaled_observation.shape, "scaled_difference", scaled_difference.shape, "diff", difference.shape, c.shape)
        # print("batch", self.batch_size, "scaled_difference", scaled_difference, "max", self.max_y, "action", action, "obs", obs, "max_x", self.x_max)
        
        avg_scl_obs = jnp.mean(scaled_observation, axis=0)
        avg_scl_diff = jnp.mean(scaled_difference, axis=0)
        
        max_sample = jnp.max(obs, axis=0)
        scaled_max  = jnp.clip((max_sample - self.min_y) / (self.max_y - self.min_y), a_min=0, a_max=1)
        
        avg_imp = jnp.mean(scaled_observation - self.scaled_obs[-1], axis=0)
        
        # print(scaled_max.shape)
        # bb = scaled_max[0]
        new_best = jnp.maximum(0.0, max_sample - self.best_rewards[0])
        s_new_best = jnp.clip((new_best) / (self.max_y - self.min_y), a_min=0, a_max=1)
        
        # self.r_mse = 0.0
        # self.r_obs = 10.0
        # print("mse", mse, "avg_imp", avg_imp, "new_best", s_new_best, "scaled_max", scaled_max, "avg_scl_obs", avg_scl_obs)
        
        e_reward = self.r_best * scaled_max + self.r_impr * avg_imp + s_new_best * self.r_new_best + mse * -1 * self.r_mse + avg_scl_obs  * self.r_obs
        e_reward = e_reward * 10
        
        
        # e_reward = jnp.mean(-jnp.abs(difference), axis=0)
        # e_reward = jnp.mean(-jnp.abs(scaled_difference), axis=0)
        # e_reward = avg_scl_obs
        
        
        # print("mse reward", mse * -1, avg_scl_obs, e_reward)
        # print("scaled obs", scaled_observation, "scaled diff", scaled_difference)
        # print("reward", reward.shape)
        reward = jnp.squeeze(e_reward, axis=-1)
        
        # print("avg_scl_obs", avg_scl_obs.shape, "avg_scl_diff", avg_scl_diff)
        
        # jax.debug.print("max_sample {} {}", max_sample, self.state)
        
        if self.best_rewards[0] < max_sample:
            self.best_rewards = self.best_rewards.at[0].set(max_sample)
        
        
        
        # print("avg_scl_obs", avg_scl_obs.shape, "avg_scl_diff", avg_scl_diff.shape)
        
        
        
        
        
        
        self.raw_rewards.append(reward)
        self.scaled_obs.append(avg_scl_obs)
        self.scaled_diff.append(avg_scl_diff)
        self.eval_obs.append(scaled_observation)
        
        self.mse.append(mse)

        

        done = False  # Episodes do not naturally end; only truncated.
        truncated = self.tick >= self.max_episode_steps
        info = {}
        
        
        

        if truncated:
            info["final_observation"] = obs
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
            # for i in self.scaled_obs[1:]:
            #     print(i.shape)
            # print(len(self.scaled_obs))
            # print(len(self.scaled_diff))
                
            # print("scaled_obs", self.scaled_obs[1:])
            
            info["scaled_obs"] = jnp.mean(jnp.array(self.scaled_obs[2:]), axis=0)
            info["best_rewards"] = self.best_rewards
            info["success"] = ((self.best_rewards[0] - self.min_y) / (self.max_y - self.min_y)) > 0.9
            info["actions"] = self.actions
            info["eval_scaled_diff"] = jnp.array(self.eval_obs)
            info["max_x"] = self.max_x
            # info["max_y"] = jnp.reshape(self.max_y, (1,1))
            # Reset tick and rewards for the next episode.
            self.tick = 0
            self.raw_rewards = []
            self.tick = 0
            self.scaled_rewards = []
            self.best_rewards = []
            self.scaled_diff = []
            self.scaled_obs = []
            self.mse = []
            self.actions = []
            self.eval_obs = []
            # jax.debug.print("maxy {}\nminy {}\nls_diff {}\nl_obs {}\ns_diff {} {}\ns_obs {} {}\n scaled_diff{} \nscaled_obs {}\n", 
            #             self.max_y, self.y_min, info["last_scaled_diff"], info["last_scaled_obs"], info["scaled_diff"], difference, info["scaled_obs"], self.state[:self.batch_size, :]
            #             ,jnp.array(self.scaled_diff), jnp.array(self.scaled_obs))
            
        # comb = np.concatenate([self.x, self.state], axis=-1)
        # print("we should not", np.array([self.tick], dtype=np.int32).shape)
        observation = {
            "actions": np.array(action, dtype=np.float32),
            "observations": np.array(obs, dtype=np.float32),
            "reward": np.array(e_reward, dtype=np.float32),
            "mask": np.array(jnp.expand_dims(self.batch_size, axis=-1), dtype=np.int32),
            "step": np.array([self.tick], dtype=np.int32)

        }
        # actions = np.array(action, dtype=np.float32)
        # observations = np.array(obs, dtype=np.float32)
        # rewards = np.array(e_reward, dtype=np.float32)
        # observations = np.full_like(observations, actions[0,0])
        # rewards = np.full_like(rewards, actions[0,0])
        # mask = np.array(jnp.expand_dims(self.batch_size, axis=-1), dtype=np.int32)
        
        # # print("actions", actions, "observations", observations, "rewards", rewards, "mask", mask)
        # observation = {
        #     "actions": actions,
        #     "observations": observations,
        #     "reward": rewards,
        #     "mask": mask

        # }

        
      
            
        return observation, reward, done, truncated, info

    def compute_y(self, x):
        x = np.atleast_2d(x)
        result = np.full(x.shape[0], self.c)

        # Add large peak
        for i in range(self.action_dim):
            result += self.A0 * np.cos(self.B0[i] * (x[:, i] - self.s0[i]))

        # Add small oscillations
        for i in range(self.action_dim):
            for k in range(self.num_oscillations):
                diff = x[:, i] - self.small_shift[i, k]
                result += self.small_A[i, k] * np.cos(self.small_B[i, k] * diff + self.small_phase[i, k])

        return result.squeeze()
    
    def find_minimum(self):
        
        num = int(np.sqrt(self.grid_resolution))
        x1_vals = np.linspace(self.x_range[0], self.x_range[1], num)
        x2_vals = np.linspace(self.x_range[0], self.x_range[1], num)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        grid_points = np.column_stack([X1.ravel(), X2.ravel()])
        y_vals = self.compute_y(grid_points)
        min_idx = np.argmin(y_vals)
        self.empirical_min_x = grid_points[min_idx]
        self.empirical_min_y = y_vals[min_idx]

    def map_to_bounds(self, actions):
        shaped_action = jnp.reshape(actions, (self.max_batches, self.action_dim))
        # shaped_action = jnp.reshape(actions, (self.max_size, self.action_dim))
        lower, upper = self.x_range[0], self.x_range[1]  # Extract lower and upper bounds
        scale = (upper - lower) / 2  # Scaling factor
        shift = (upper + lower) / 2  # Shift factor
        c = scale * shaped_action + shift
        return c
    

    def render(self):
        """
        Print the current state and polynomial parameters.
        """
        print(f"Current x: {self.actions}\n")
        
        
