# multi_function_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
import jax


from src.tasks.envs.function_envs.sampling_functions.branin_sampler import BraninSampler
from src.tasks.envs.function_envs.sampling_functions.eggholder_sampler import EggholderSamplerND
from src.tasks.envs.function_envs.sampling_functions.ackley_sampler import AckleySampler
from src.tasks.envs.function_envs.sampling_functions.cosine_sampler import CosineSampler
from src.tasks.envs.function_envs.sampling_functions.poly_sampler import PolySampler
from src.tasks.envs.function_envs.sampling_functions.rosenbrock_sampler import RosenbrockSampler
from src.tasks.envs.function_envs.sampling_functions.michalewicz_sampler import MichalewiczSampler
from src.tasks.envs.function_envs.sampling_functions.hartmann_sampler import Hartmann6Sampler


class MultiFunctionEnv(gym.Env):
    """
    Optimization environment that can sample from multiple function types.

    The environment interacts with a function sampled at reset time from a
    specified list of function types (e.g., 'ackley', 'cosine', 'poly').
    """
    def __init__(self, env_config):
        super().__init__()

        # --- Core Environment Config ---
        self.x_range = env_config["bounds"]
        self.total_samples = env_config["total_episode_samples"]
        # self.max_episode_steps = env_config["max_episode_steps"] # This will be calculated
        self.batches = env_config["batches"] # List of possible batch sizes, e.g., [1, 5, 10]
        self.max_batches = env_config["max_batches"] # Max possible samples per step
        self.action_dim = env_config["action_dim"]
        self.use_random_action_on_reset = env_config.get("random_action_on_reset", False) # Use random init action?
        self.use_random_action_on_step = env_config.get("random_action_on_step", False) # Use random action every step?

        # --- Function Sampler Config ---
        # List of function names to sample from, e.g., ['ackley', 'cosine']
        self.function_types = env_config["function_types"]
        if not self.function_types:
            raise ValueError("env_config must contain a non-empty list 'function_types'")

        # Optional configurations specific to each function type
        self.function_configs = env_config.get("function_configs", {})
       
        # Instantiate the samplers
        self.samplers = {}
        for func_name in self.function_types:
            sampler_class = get_sampler_class(func_name)
            # Pass specific config if available, else None
            sampler_config = self.function_configs.get(func_name, None)
            # print(f"Initializing {func_name} with config: {sampler_config}") # Debug print
            self.samplers[func_name] = sampler_class(self.action_dim, self.x_range, sampler_config)

        # --- State Variables (initialized in reset) ---
        self.current_sampler = None # The sampler instance chosen for the current episode
        self.current_compute_y = None # Direct reference to the compute_y method for efficiency
        self.optimum_point = None   # Optimum point for the current function
        self.min_y = None           # Min value for the current function
        self.max_y = None           # Max value for the current function (value at optimum)
        self.batch_size = None      # Actual batch size for the current step/episode
        self.tick = 0               # Step counter within the episode
        self.achieved_success_threshold = False # If high reward was achieved last episode
        self.best_scaled_y_so_far = -np.inf # Track best scaled y across steps
        self.best_obs_x = None # Store best observation x (if needed)

        # --- Calculated Episode Length ---
        # Ensure max_batches is used for calculation if batch size varies
        self.max_episode_steps = self.total_samples // self.max_batches
        if self.total_samples % self.max_batches != 0:
             self.max_episode_steps += 1 # Account for remaining samples


        # --- Reward Scaling Parameters ---
        # You might want to make these configurable via env_config too
        self.r_scale = env_config.get("r_scale", 10.0)
        self.r_best = env_config.get("r_best", 0.8)        # Weight for the best sample in the current batch
        self.r_impr = env_config.get("r_impr", 0.1)        # Weight for average improvement over last step
        self.r_avg = env_config.get("r_avg", 0.0)          # Weight for average value in the current batch (use r_obs instead?)
        self.r_new_best = env_config.get("r_new_best", 0.1)# Bonus for exceeding the best *ever* found in the episode
        self.r_mse = env_config.get("r_mse", 0.0)          # Penalty based on MSE from the maximum (self.max_y)
        self.r_obs = env_config.get("r_obs", 0.0)          # Weight for the average scaled observation value
        self.b_pen = env_config.get("b_pen", 0.0)          # Penalty for actions near bounds (optional, not implemented here)
        self.r_suc = env_config.get("r_suc", 3.0)          # Bonus for achieving success threshold at the end
        self.success_threshold = env_config.get("success_threshold", 0.95) # Scaled y value considered success
        
        

        # Define action and observation spaces using max_batches
        self.action_space = spaces.Box(
            low=np.full((self.max_batches, self.action_dim), -1.0), # Standard practice: actions in [-1, 1]
            high=np.full((self.max_batches, self.action_dim), 1.0),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict({
             # Previous actions (mapped to bounds)
            "actions": spaces.Box(
                low=np.full((self.max_batches, self.action_dim), self.x_range[0]),
                high=np.full((self.max_batches, self.action_dim), self.x_range[1]),
                dtype=np.float32
            ),
             # Observations (function values y) corresponding to previous actions
            "observations": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_batches, 1),
                dtype=np.float32
            ),
             # Reward obtained from the previous step
            "reward": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(1,),
                dtype=np.float32
            ),
             # Mask indicating the actual number of valid samples (batch_size)
            "mask": spaces.Box(
                low=0, high=self.max_batches, # Represents the count
                shape=(1,),
                dtype=np.int32
            ),
             # Current step number in the episode
            "step": spaces.Box(
                low=0, high=self.max_episode_steps,
                shape=(1,),
                dtype=np.int32
            )
        })

        # Internal state for reward calculation and info dict
        self._last_avg_scaled_obs = 0.0
        self._episode_raw_rewards = []
        self._episode_mses = []
        self._episode_actions = []
        self._episode_eval_obs = [] # Store raw observations for analysis
        self._episode_regret = []
        self._episode_actions = []
        
        self._select_and_initialize_function() # Initialize the function sampler for the first episode


    def _select_and_initialize_function(self):
        """Randomly selects a function type and initializes it."""
        chosen_sampler_name = np.random.choice(self.function_types)
        self.current_sampler = self.samplers[chosen_sampler_name]
        # print(f"Resetting with function: {chosen_sampler_name}") # Debug print

        # Initialize the chosen sampler to get its specific parameters
        self.optimum_point, self.min_y, self.max_y = self.current_sampler.initialize()

        # Check for invalid bounds
        if self.max_y <= self.min_y:
             print(f"Warning: max_y ({self.max_y}) <= min_y ({self.min_y}) for {chosen_sampler_name}. Adjusting min_y slightly.")
             # Attempt recovery, though this indicates an issue in sampler's bounds calculation
             self.min_y = self.max_y - 1e-6
             # Or raise an error:
             # raise ValueError(f"Initialization failed for {chosen_sampler_name}: max_y ({self.max_y}) <= min_y ({self.min_y})")


        # Store direct reference to compute_y for faster access in step
        self.current_compute_y = self.current_sampler.compute_y
        # print(f"  Optimum: {self.optimum_point}, MinY: {self.min_y:.4f}, MaxY: {self.max_y:.4f}") # Debug


    def _scale_observation(self, obs):
        """Scales observation y to be roughly in [0, 1]."""
        # Avoid division by zero if max_y == min_y (should be handled in init)
        denominator = self.max_y - self.min_y
        # # if denominator < 1e-9: # Use a small tolerance
        # #      # If max == min, all scaled values should be conceptually the same.
        # #      # Return 0.5 or 1.0 depending on interpretation. Let's use 1.0 if obs == max_y.
        #      return         np.where(np.isclose(obs, self.max_y), 1.0, 0.0).astype(np.float32)
        
        

        scaled = (obs - self.min_y) / denominator
        
        return jnp.clip(scaled, 0.0, 1.0) # Use jnp for consistency if using JAX downstream
    

    def map_to_bounds(self, actions):
        """Maps actions from [-1, 1] to the environment's x_range."""
        # Ensure actions are numpy arrays for this mapping
        actions = np.asarray(actions)
        lower, upper = self.x_range
        scale = (upper - lower) / 2.0
        shift = (upper + lower) / 2.0
        return scale * actions + shift

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility if seed is used

        # Reset internal episode state
        self.tick = 0
        self._last_avg_scaled_obs = 0.0
        self.best_scaled_y_so_far = -np.inf
        self.best_obs_x = None # Store best observation x (if needed)  
        self._episode_raw_rewards = []
        self._episode_mses = []
        self._episode_actions = []
        self._episode_eval_obs = [] # Reset observation storage
        self._episode_regret = [] # Reset regret storage
        self._episode_actions = [] # Reset actions storage

        # Re-initialize function ONLY if success was achieved previously,
        # OR if it's the very first reset (self.current_sampler is None)
        # This allows continuing optimization on the same function if it wasn't solved.
        # You might want to always re-initialize, depending on the training paradigm.
        # if self.achieved_success_threshold or self.current_sampler is None:
        # Always re-initialize function on reset as per original request
        
        # print("we are no longer resetting the function") # Debug print
        # self._select_and_initialize_function()
        self.achieved_success_threshold = False # Reset success flag

        # --- Initial Action and Observation ---
        # Determine batch size for this episode/step (can vary)
        self.batch_size = np.random.choice(self.batches) if len(self.batches) > 1 else self.batches[0]
        # Ensure batch_size doesn't exceed max_batches
        self.batch_size = min(self.batch_size, self.max_batches)


        # Generate initial action (either zeros or random in [-1, 1])
        
        if self.use_random_action_on_reset:
             # Random actions in the [-1, 1] space
            initial_actions_normalized = self.np_random.uniform(-1.0, 1.0, size=(self.max_batches, self.action_dim)).astype(np.float32)
        else:
             # Zero actions (center of the space before mapping)
            initial_actions_normalized = np.zeros((self.max_batches, self.action_dim), dtype=np.float32)

        # Map initial actions to the actual function bounds
        initial_actions_mapped = self.map_to_bounds(initial_actions_normalized)

        # Compute initial observations (y values)
        # Use only the first `batch_size` actions for computation
        just_for_test = self.current_compute_y(initial_actions_mapped)
        
        obs_raw = self.current_compute_y(initial_actions_mapped[:self.batch_size, :])
        obs_raw = np.atleast_1d(obs_raw) # Ensure it's at least 1D

        # Pad observations to max_batches size (e.g., with NaNs or zeros)
        # Padding with a value indicating invalidity (like NaN or -inf) is often better.
        # Using min_y might be reasonable here.
        # padded_obs_raw = np.full((self.max_batches, 1), self.min_y, dtype=np.float32)
        padded_obs_raw = np.full((self.max_batches, 1), 0, dtype=np.float32)
        padded_obs_raw[:self.batch_size, 0] = obs_raw

        # Scale the valid observations
        scaled_observation = self._scale_observation(obs_raw) # Only scale the valid ones
        
        max_index = np.argmax(scaled_observation)
        self.best_obs_x = initial_actions_mapped[max_index] # Store the best observation x (if needed)
        

        # Calculate metrics based on valid observations
        current_best_scaled_y = jnp.clip(jnp.max(scaled_observation), a_min=0.0, a_max=1.0) 
        avg_scaled_obs = jnp.clip(jnp.mean(scaled_observation), a_min=0.0, a_max=1.0) 

        # Update overall best for the episode
        self.best_scaled_y_so_far = float(current_best_scaled_y)

        # Calculate initial reward (no improvement term yet)
        # Reward for initial state might be zero or based on initial sample quality
        reward = (self.r_best * current_best_scaled_y + self.r_obs * avg_scaled_obs) * self.r_scale
        reward = jnp.array([reward], dtype=jnp.float32) # Ensure shape (1,)

        # Store initial state for reward calculation in the next step
        self._last_avg_scaled_obs = float(avg_scaled_obs)
        # Store actions and observations for info dict
        self._episode_actions.append(np.array(initial_actions_mapped)) # Store mapped actions
        self._episode_eval_obs.append(np.array(padded_obs_raw)) # Store raw obs
        # print("dsfg", np.array(np.expand_dims(just_for_test, axis=-1), dtype=np.float32).shape)
        # print("dsfg", np.reshape(np.array(just_for_test, dtype=np.float32), (self.max_batches, 1)).shape)

        # Construct initial observation dictionary
        observation = {
            # "actions": np.array(initial_actions_mapped, dtype=np.float32), # Action that LED to obs
            # "observations": np.array(padded_obs_raw, dtype=np.float32),    # Resulting raw Y values
            "actions": np.array(initial_actions_normalized, dtype=np.float32), # Action that LED to obs
            "observations": np.reshape(np.array(just_for_test, dtype=np.float32), (self.max_batches, 1)), # Raw Y values resulting from action
            "reward": np.array(reward, dtype=np.float32),                 # Reward for this state transition (0 for first step)
            "mask": np.array([self.batch_size], dtype=np.int32),          # Valid samples in "observations"
            "step": np.array([self.tick], dtype=np.int32)                 # Starts at 0
        }
        # print(f"Reset Obs: MaxS={current_best_scaled_y:.3f}, AvgS={avg_scaled_obs:.3f}, Rew={reward[0]:.3f}") # Debug
        return observation, {} # Return observation and empty info dict


    def step(self, action):
        if self.current_compute_y is None:
            raise RuntimeError("Environment must be reset before stepping.")

        self.tick += 1

        # Action comes from the agent (assumed in [-1, 1])
        # Ensure action has the correct shape (max_batches, action_dim)
        action = np.asarray(action).reshape((self.max_batches, self.action_dim))

        # --- Action Processing ---
        # Option to override agent's action with random exploration
        if self.use_random_action_on_step:
             action_normalized = self.np_random.uniform(-1.0, 1.0, size=(self.max_batches, self.action_dim)).astype(np.float32)
        else:
             action_normalized = action # Use the agent's action

        # Map action from [-1, 1] to function bounds [low, high]
        action_mapped = self.map_to_bounds(action_normalized)
        

        # --- Observation Calculation ---
        # Decide batch size for this step (could be fixed or variable)
        # For now, let's keep it fixed per episode, decided in reset. Re-randomizing here is also possible.
        # self.batch_size = np.random.choice(self.batches) if len(self.batches) > 1 else self.batches[0]
        # self.batch_size = min(self.batch_size, self.max_batches)
        just_for_test = self.current_compute_y(action_mapped)
        # Compute function values (y) for the first `batch_size` actions
        obs_raw = self.current_compute_y(action_mapped[:self.batch_size, :])
        # ss = self.current_compute_y(self.optimum_point)
        # jax.debug.print("action {} obs {} \noptimu {} ops {}\noptimu {} oss{}\nmin {} obs {} max {}\n", 
        #                 action_mapped, obs_raw, self.optimum_point, self.max_y, self.optimum_point, ss, self.min_y, obs_raw, self.max_y)
        
        obs_raw = np.atleast_1d(obs_raw)

        # Pad observations to max_batches size
        padded_obs_raw = np.full((self.max_batches, 1), self.min_y, dtype=np.float32)
        padded_obs_raw[:self.batch_size, 0] = obs_raw
        
        # print(obs_raw, padded_obs_raw, self.min_y, self.max_y)

        # --- Reward Calculation ---
        # Scale the valid observations
        scaled_observation = self._scale_observation(obs_raw) # Shape: (batch_size,)

        # Calculate metrics from valid observations
        current_best_scaled_y = jnp.max(scaled_observation) 
        avg_scaled_obs = jnp.mean(scaled_observation)
        
        max_index = np.argmax(scaled_observation)
        self.best_obs_x = action_mapped[max_index] # Store the best observation x (if needed)

        # Improvement metrics
        avg_improvement = avg_scaled_obs - self._last_avg_scaled_obs
        new_best_bonus = jnp.maximum(0.0, current_best_scaled_y - self.best_scaled_y_so_far) # Improvement over episode best

        # MSE from maximum (optional penalty)
        # Ensure difference calculation uses raw values before scaling
        difference_from_max = self.max_y - obs_raw
        
        
        
        mse = jnp.mean(jnp.square(difference_from_max)) 
        
        scaled_difference = jnp.clip(difference_from_max / (self.max_y - self.min_y), 0.0, 1.0) # Scale to [0, 1]
        
        # jax.debug.print("min {} < obs {} < max {}\nscaled {}",self.min_y, obs_raw, self.max_y, scaled_difference)
        
        

        # Check for success threshold achievement *using the best in this step*
        success_achieved_this_step = current_best_scaled_y >= self.success_threshold

        # --- Termination and Truncation ---
        terminated = False # Usually False in optimization unless a target is hit exactly?
        truncated = self.tick >= self.max_episode_steps

        # Success bonus (only if truncated and threshold met)
        success_bonus = 0.0
        if truncated and success_achieved_this_step:
             success_bonus = self.r_suc
             self.achieved_success_threshold = True # Mark for potential function reset
             
             
        # jax.debug.print("max_act {} action {}\nmax_y {}  obs {}\nmax_y {} min_y {}\nscaled_diff: \n{}\n", 
        #                 self.optimum_point, action_mapped, self.max_y, obs_raw, self.max_y, self.min_y, scaled_difference)

        # Combine reward components
        reward = (
            self.r_best * current_best_scaled_y + # Quality of best sample this step
            self.r_impr * avg_improvement +       # Average improvement over last step
            self.r_new_best * new_best_bonus +    # Bonus for new episode best
            self.r_obs * avg_scaled_obs +         # Quality of average sample this step
           -self.r_mse * mse +                   # Penalty for distance from max (optional)
            success_bonus                         # End-of-episode success bonus
        ) * self.r_scale

        reward = jnp.array([reward], dtype=jnp.float32) # Ensure shape (1,)

        # Update internal state for next step
        self._last_avg_scaled_obs = float(avg_scaled_obs)
        if float(current_best_scaled_y) > self.best_scaled_y_so_far:
            self.best_scaled_y_so_far = float(current_best_scaled_y)
            
        # print("sadg", action_mapped.shape, action.shape, padded_obs_raw.shape, scaled_observation.shape)
        # Store step data
        self._episode_raw_rewards.append(float(reward[0])) # Store scalar reward
        self._episode_mses.append(float(mse))
        self._episode_actions.append(np.array(action_mapped)) # Store mapped actions taken
        self._episode_eval_obs.append(np.array(padded_obs_raw)) # Store raw obs received
        self._episode_regret.append(np.array(scaled_difference)) # Store regret (scaled difference from max)
        self._episode_actions.append(np.array(action_mapped)) # Store mapped actions taken

        # --- Construct Observation for Next State ---
        # print(np.array(padded_obs_raw, dtype=np.float32).shape, np.array(np.expand_dims(scaled_observation, axis=-1)).shape, just_for_test.shape)
        
        observation = {
            # "actions": np.array(action_mapped, dtype=np.float32),   # Action that led to this state
            # "observations": np.array(padded_obs_raw, dtype=np.float32), # Raw Y values resulting from action
            "actions": np.array(action, dtype=np.float32),   # Action that led to this state
            "observations": np.reshape(np.array(just_for_test, dtype=np.float32), (self.max_batches, 1)), # Raw Y values resulting from action
            "reward": np.array(reward, dtype=np.float32),          # Reward obtained for reaching this state
            "mask": np.array([self.batch_size], dtype=np.int32),   # Valid samples in "observations"
            "step": np.array([self.tick], dtype=np.int32)          # Current step number
        }

        # --- Info Dictionary (populated on truncation) ---
        info = {}
        if truncated:
            info["final_observation"] = padded_obs_raw # Last raw observation batch
            info["episode_length"] = self.tick
            info["reward_per_episode"] = np.sum(self._episode_raw_rewards)
            info["rewards"] = self._episode_raw_rewards
            # info["rewards"] = np.array(self._episode_raw_rewards) # Optional: full list of rewards
            info["batch_mse"] = np.mean(self._episode_mses)
            info["last_avg_scaled_obs"] = avg_scaled_obs
            info["best_rewards"] = np.array(self.best_scaled_y_so_far)
            info["scaled_diff"] = np.mean(self._episode_regret) # Average regret (scaled difference from max)
            info["last_scaled_diff"] = np.array(scaled_difference)
            info["actions"] = self._episode_actions
            # info["actions_episode"] = np.stack(self._episode_actions) # Optional: full action history
            # info["observations_episode"] = np.stack(self._episode_eval_obs) # Optional: full observation history
            info["success"] = np.array(self.achieved_success_threshold)
            info["max_x"] = self.optimum_point # Optimum of the function used
            info["distance_from_max"] = np.linalg.norm(self.best_obs_x - self.optimum_point) # Distance from optimum

            # Optionally reset internal lists here if memory is a concern,
            # but they are reset in reset() anyway.

        # print(f"Step {self.tick}: BestS={current_best_scaled_y:.3f}, AvgS={avg_scaled_obs:.3f}, Impr={avg_improvement:.3f}, NewB={new_best_bonus:.3f}, MSE={mse:.3f}, Rew={reward[0]:.3f}") # Debug

        return observation, float(reward[0]), terminated, truncated, info

    def render(self):
        # Basic render function
        print(f"Step: {self.tick}")
        print(f"  Current Best Scaled Y: {self.best_scaled_y_so_far:.4f}")
        print(f"  Last Avg Scaled Obs: {self._last_avg_scaled_obs:.4f}")
        if self.current_sampler:
             print(f"  Current Function Type: {self.current_sampler.__class__.__name__}")
             # print(f"  Optimum Point: {self.optimum_point}") # Can be verbose

    def close(self):
        # Add any necessary cleanup here
        pass
    
    
    
def get_sampler_class(name):
    if name == 'ackley':
        return AckleySampler
    elif name == 'cosine':
        return CosineSampler
    elif name == 'poly':
        return PolySampler
    elif name == 'eggholder':
        return EggholderSamplerND
    elif name == 'rosenbrock':
        return RosenbrockSampler
    elif name == 'michalewicz':
        return MichalewiczSampler
    elif name == 'hartmann6':
        return Hartmann6Sampler
    elif name == 'branin':
        return BraninSampler
        
   
    else:
        raise ValueError(f"Unknown function sampler name: {name}")