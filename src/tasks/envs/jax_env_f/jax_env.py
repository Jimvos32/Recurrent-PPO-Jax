import jax
import jax.numpy as jnp
import chex
import flax.struct as struct # Or use standard dataclasses
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Dict, Any

from jax.experimental import checkify


# from src.tasks.envs.jax_env_f.jax_function_samplers import initialize_sampler_dispatch, compute_y_sampler_dispatch, EnvParams
from src.tasks.envs.jax_env_f.jax_disp_samplers import initialize_sampler_dispatch, compute_y_sampler_dispatch, EnvParams
# from src.tasks.envs.jax_env_f.jax_disp_samplers import EnvParams, initialize_sampler_dispatch, compute_y_sampler_dispatch
# from src.tasks.envs.jax_env.env_params import EnvParams
# from 

# Assume JAX versions of your samplers exist:
# from .jax_samplers import initialize_sampler, compute_y_sampler # You'll need to create these


@struct.dataclass
class EnvState:
    """Dynamic environment state."""
    tick: int
    # --- Sampler State ---
    sampler_type_index: int # Which function is active (e.g., 0 for Ackley)
    optimum_point: chex.Array # Shape (action_dim,)
    min_y: float
    max_y: float
    
    # Any other sampler-specific parameters returned by initialize_sampler if needed
    # --- Episode Tracking ---
    last_avg_scaled_obs: float
    best_scaled_y_so_far: float
    best_obs_x_so_far: chex.Array # Shape (action_dim,)
    batch_size: int
    achieved_success: bool
    # --- Last Step Info (for constructing next obs) ---
    last_action_mapped: chex.Array # Shape (max_batches, action_dim)
    last_padded_obs_raw: chex.Array # Shape (max_batches, 1)
    last_action: chex.Array # Shape (max_batches, action_dim)
    last_raw_obs: chex.Array # Shape (max_batches, 1)
    last_reward: float
    done: bool # True if episode is done (for info dict)
    max_steps_in_episode: int # Set during reset, used for truncation
    params_for_compute: Dict[str, Any] # Parameters for compute_y_sampler
    # --- JAX key ---
    
    episode_counter: int # Counter for episode tracking (if needed)
    key: chex.PRNGKey

 
class MultiFunctionGymnax(environment.Environment):
    
    metadata = {
        "render_modes": [],  # Specify supported render modes (empty list if none)
        "render_fps": 0,     # Specify render frame rate (0 if not applicable)
    }


    def __init__(self):
        """Use default_params for instantiation."""
        super().__init__()
        

    @property
    def default_params(self) -> EnvParams:
        # Define the default parameters here
        return EnvParams()
    
    def initialize(
        self, key: chex.PRNGKey, params: EnvParams, action_dim: int, max_batches: int
    ) -> Tuple[chex.ArrayTree, EnvState]:
        """Initializes the environment."""
        # return initialize_sampler(key, params, action_dim, max_batches)
        return initialize_sampler_dispatch(key, params, action_dim, max_batches)
    
    @staticmethod
    def step_env(
        key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams, max_batches: int, action_dim: int
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, Dict]:
        """Performs one step in the environment."""
        key, key_step, key_random_action = jax.random.split(key, 3)

        # --- Action Processing ---
        # Option to override agent's action with random exploration
        
        action_normalized = action.reshape(max_batches, action_dim)
        
        # idx = state.sampler_type_index.astype(int)
        
        # f_name = FuncIndices._value2member_map_.get(idx).name
        range = state.params_for_compute['common']["bounds"]
        action_mapped = map_to_bounds_jax(action_normalized, range)

        # --- Observation Calculation ---
        # Use the current batch_size from state
        # batch_size = state.batch_size
        # Compute function values using JAX dispatch based on state.sampler_type_index
        # sampler_params_for_compute = { # Pass necessary params from state
        #     "optimum_point": state.optimum_point,
        #     "min_y": state.min_y,
        #     "max_y": state.max_y,
        #     "type_index": state.sampler_type_index
        # }
        
        # bb = 1
        # print("here is something that is not corret!!!!!!!")
        # obs_raw = compute_y_sampler(action_mapped, state.params_for_compute, params)
        obs_raw = compute_y_sampler_dispatch(action_mapped, state.params_for_compute, params)
        
        
        
        # jax.debug.print("action {} test_obs {}\n obs_raw {} \ntick {}", action_mapped, test_obs, obs_raw, state.tick)
        obs_raw = jnp.atleast_1d(obs_raw)
        
        mask = jnp.arange(max_batches) < state.batch_size
        
        # print("obs_raw", obs_raw.shape, action_mapped.shape, state.params_for_compute, params)

        # # Pad observations
        # # Use jnp.nan or state.min_y for padding? Using min_y as in original.
        # padded_obs_raw = jnp.full((max_batches, 1), state.min_y, dtype=jnp.float32)
        
        # padded_obs_raw = obs_raw.at[state.batch_size:].set(jnp.zeros_like(obs_raw[batch_size:])) # i thought this was fine since the output will always be the same shape
        # # padded_obs_raw = padded_obs_raw.at[:batch_size, 0].set(obs_raw) # this was the original code you provided which gave similar issues with tracing

        # --- Reward Calculation ---
        # Scale the valid observations
        scaled_observation = scale_observation_jax(obs_raw, state.min_y, state.max_y) # Shape: (batch_size,)
        
        # jax.debug.print("state min {} {}\n state max {} {}", state.min_y, state.params_for_compute["common"]["min_y"], state.max_y, state.params_for_compute["common"]["max_y"])

        # jax.debug.print("scaled_observation {} mask {} obs_raw {}\nmin {}\nmax {}\naction {}\n", scaled_observation, mask, obs_raw, state.min_y, state.max_y, action)
        
        # Calculate metrics
        current_best_scaled_y = jnp.max(jnp.where(mask, scaled_observation, -jnp.inf)) # Handle empty case
        sum_scaled_obs = jnp.sum(jnp.where(mask, scaled_observation, 0.0))
        avg_scaled_obs = sum_scaled_obs / state.batch_size # Average over valid observations

        # Update best_obs_x for the episode
        best_idx_in_batch = jnp.argmax(jnp.where(mask, scaled_observation, -jnp.inf))
        current_best_x = action_mapped[best_idx_in_batch] # X corresponding to best Y *in this batch*

        # Improvement metrics
        avg_improvement = avg_scaled_obs - state.last_avg_scaled_obs
        new_best_bonus = jnp.maximum(0.0, current_best_scaled_y - state.best_scaled_y_so_far)

        # MSE from maximum (optional penalty)
        difference_from_max = state.max_y - obs_raw
        sum_sq_diff = jnp.sum(jnp.where(mask, jnp.square(difference_from_max), 0.0))
        mse = sum_sq_diff / jnp.maximum(state.batch_size, 1)

        # Scaled difference for regret calculation
        scaled_difference = scale_observation_jax(difference_from_max + state.min_y, state.min_y, state.max_y)

        
        
        
        # Success threshold achievement
        success_achieved_this_step = current_best_scaled_y >= params.success_threshold
        achieved_success_updated = state.achieved_success | success_achieved_this_step
        
        
        # --- Termination and Truncation ---
        tick = state.tick + 1
        terminated = False # Usually false
        # truncated = tick >= state.max_steps_in_episode # Check if max steps reached
        truncated = success_achieved_this_step | tick >= state.max_steps_in_episode# Check if max steps reached

        # Success bonus (only if truncated and threshold met this step or previously)
        # Original logic adds bonus if truncated AND success_achieved_this_step. Let's stick to that.
        success_bonus = jax.lax.select(
            success_achieved_this_step,
            params.r_suc,
            0.0
        )
        
        reward = (
            params.r_step_cost +
            params.r_impr * avg_improvement +
            success_bonus
        )
        
        # jax.debug.print("reward {} step_cost {} impr {} bonus {} tick {} trunc {}\n obs {} imp {} scale {} end {}",
        #                 reward, params.r_step_cost, params.r_impr * avg_improvement, success_bonus, tick, truncated,
        #                 scaled_observation, avg_improvement, params.r_impr, params.r_impr * avg_improvement)
        

        # Combine reward components
        # reward = (
        #     params.r_best * current_best_scaled_y +
        #     params.r_impr * avg_improvement +
        #     params.r_new_best * new_best_bonus +
        #     params.r_obs * avg_scaled_obs +
        #     -params.r_mse * mse +
        #     success_bonus
        # ) * params.r_scale
        
        
        # jax.debug.print("reward {} best_y {} avg_obs {} new_best {} mse {} success_bonus {} impr {}\n",
        #                 reward, current_best_scaled_y, avg_scaled_obs, new_best_bonus, mse, success_bonus, avg_improvement)
        
        # jax.debug.print("reward =  {} best {} + impr {} + new best {} + obs {} + mse {} + {}bonuses \nscaled {} curr_best {} sr {}\n",
        #                 reward, params.r_best * current_best_scaled_y, params.r_impr * avg_improvement, params.r_new_best * new_best_bonus,
        #     params.r_obs * avg_scaled_obs, -params.r_mse * mse, success_bonus, scaled_observation, current_best_scaled_y, params.r_best)

        # print("reward", params)
        # --- Update State ---
        # Update overall best y and corresponding x
        best_scaled_y_updated = jnp.maximum(state.best_scaled_y_so_far, current_best_scaled_y)
        # Update best_x only if the current batch produced a better y
        best_obs_x_updated = jax.lax.select(
            current_best_scaled_y > state.best_scaled_y_so_far,
            current_best_x,
            state.best_obs_x_so_far
        )
        
        done = terminated | truncated

        # state = EnvState(
        #     tick=tick,
        #     sampler_type_index=state.sampler_type_index, # Remains same for episode
        #     optimum_point=state.optimum_point,
        #     min_y=state.min_y,
        #     max_y=state.max_y,
        #     last_avg_scaled_obs=avg_scaled_obs,
        #     best_scaled_y_so_far=best_scaled_y_updated,
        #     best_obs_x_so_far=best_obs_x_updated,
        #     batch_size=state.batch_size, # Remains same for episode
        #     achieved_success=achieved_success_updated, # Mark if ever achieved
        #     last_action_mapped=action_mapped, # Store action taken
        #     last_padded_obs_raw=padded_obs_raw, # Store resulting obs
        #     last_reward=reward,
        #     done=truncated, # Mark if episode is done
        #     max_steps_in_episode=state.max_steps_in_episode, # Set max steps for this episode
        #     params_for_compute=state.params_for_compute, # Pass params for compute_y_sampler
        #     key=key # Pass key for next step
        # )
        
        
        # print("asgd", obs_raw.shape, action_mapped.shape, scaled_observation.shape, jnp.squeeze(scaled_observation).shape)
        last_obs = jnp.reshape(scaled_observation, (max_batches, 1))
        last_raw = jnp.reshape(obs_raw, (max_batches, 1))
        
        
        
        
        state = state.replace( # Use replace for immutability with dataclasses/Pytrees
            tick=tick,
            # sampler_type_index, optimum_point, min_y, max_y, batch_size, params_for_compute remain same
            last_avg_scaled_obs=avg_scaled_obs,
            best_scaled_y_so_far=best_scaled_y_updated,
            best_obs_x_so_far=best_obs_x_updated,
            achieved_success=achieved_success_updated,
            last_action_mapped=action_mapped, # Store full mapped actions
            last_padded_obs_raw=last_obs, # Store padded obs)
            last_action=action_normalized, # Store action taken
            last_raw_obs=last_raw, # Store raw observation
            last_reward=reward,
            done=done, # Store done flag reflecting current step's outcome
            # max_steps_in_episode remains same
            key=key # Pass updated key if state carries it
        )

        # --- Construct Observation and Info ---
        obs = get_obs(state, params)
        
        # jax.debug.print("tick {} max_steps_in_episode {} {} {} {}", tick, state.max_steps_in_episode, done, terminated, truncated)

        
        # print
        
        info = get_info(state, params, done, jnp.mean(scaled_difference)) # Pass done flag
        
        obs = get_obs(state, params)
        
        # jax.debug.print("obs {} action {} reward {} mask {} step {} min{} max {} raw {}", obs, action_mapped, reward, mask, tick, state.min_y, state.max_y, obs_raw)
        
        # jax.debug.print("last_rew {} pas_rew {}\nsucces {} action {} c_best {}\n obs{} differnce {} mask {} \nraw obs {} min {} max {}\n", 
        #                 reward, obs["reward"], success_achieved_this_step, action, current_best_scaled_y, scaled_observation, scaled_difference, mask, obs_raw, state.min_y, state.max_y)
        
        # jax.debug.print("ssdf {}", obs)

        return obs, state, reward, done, info

    @staticmethod
    def reset_env(
        key: chex.PRNGKey, params: EnvParams, action_dim: int, max_batches: int
    ) -> Tuple[chex.ArrayTree, EnvState]:
        """Resets the environment to an initial state."""
        key, key_sampler, key_batch, key_action = jax.random.split(key, 4)

        # Select function type and initialize its parameters
        sampler_type_index = jax.random.choice(key_sampler, params.function_type_indices)
        # Use JAX dispatch function for sampler initialization
        # jax.debug.print("sampler_type_index {} {} {}", sampler_type_index, params.sampler_configs["specific"].keys(), params.function_type_indices)
        # print("sampler_type_index", sampler_type_index, type(sampler_type_index))
        
        
        # sam_con = initialize_sampler(key_sampler, sampler_type_index, action_dim, params)
        sam_con = initialize_sampler_dispatch(key_sampler, sampler_type_index, params)

        
        min_y = sam_con['common']['min_y']
        max_y = sam_con['common']['max_y']
        optimum_point = sam_con['common']['optimum_point']
        
        

        # Check for invalid bounds (JAX style - maybe return valid flag?)
        # Simple check: if max_y <= min_y: print warning or adjust
        min_y = jnp.minimum(min_y, max_y - 1e-6) # Ensure min < max

        # Determine batch size for this episode
        batch_size = jax.lax.select( # Handle single batch size case
            params.batches.shape[0] > 1,
            jax.random.choice(key_batch, params.batches),
            params.batches[0]
        )
        
        max_steps = params.total_samples // batch_size # Calculate steps per batch
        
        
        # batch_size = jnp.minimum(batch_size, params.max_batches) # Clip to max

        # Generate initial actions
        initial_actions_normalized = jax.lax.cond(
            params.use_random_action_on_reset,
            lambda k: jax.random.uniform(k, (max_batches, action_dim), minval=-1.0, maxval=1.0),
            lambda k: jnp.zeros((max_batches, action_dim)),
            key_action,
        )
        initial_actions_mapped = map_to_bounds_jax(initial_actions_normalized, params.x_range)

        # Compute initial observations
        # sampler_params_for_compute = {
        #     "optimum_point": optimum_point, "min_y": min_y, "max_y": max_y,
        #     "type_index": sampler_type_index
        # }
        
        
        # obs_raw = compute_y_sampler(initial_actions_mapped, sam_con, params)
        obs_raw = compute_y_sampler_dispatch(initial_actions_mapped, sam_con, params)
        obs_raw = jnp.atleast_1d(obs_raw)
        
        mask = jnp.arange(max_batches) < batch_size
        
        # print("obs_raw", obs_raw.shape, action_mapped.shape, state.params_for_compute, params)

        # # Pad observations
        # # Use jnp.nan or state.min_y for padding? Using min_y as in original.
        # padded_obs_raw = jnp.full((max_batches, 1), state.min_y, dtype=jnp.float32)
        
        # padded_obs_raw = obs_raw.at[state.batch_size:].set(jnp.zeros_like(obs_raw[batch_size:])) # i thought this was fine since the output will always be the same shape
        # # padded_obs_raw = padded_obs_raw.at[:batch_size, 0].set(obs_raw) # this was the original code you provided which gave similar issues with tracing

        # --- Reward Calculation ---
        # Scale the valid observations
        scaled_observation = scale_observation_jax(obs_raw, min_y, max_y) # Shape: (batch_size,)

        # Calculate metrics
        current_best_scaled_y = jnp.max(jnp.where(mask, scaled_observation, -jnp.inf)) # Handle empty case
        sum_scaled_obs = jnp.sum(jnp.where(mask, scaled_observation, 0.0))
        avg_scaled_obs = sum_scaled_obs / batch_size # Average over valid observations

        # Update best_obs_x for the episode
        best_idx_in_batch = jnp.argmax(jnp.where(mask, scaled_observation, -jnp.inf))
        current_best_x = initial_actions_mapped[best_idx_in_batch] # X corresponding to best Y *in this batch*

        # # Improvement metrics
        # avg_improvement = avg_scaled_obs - state.last_avg_scaled_obs
        # new_best_bonus = jnp.maximum(0.0, current_best_scaled_y - state.best_scaled_y_so_far)

        # MSE from maximum (optional penalty)
        difference_from_max = max_y - obs_raw
        sum_sq_diff = jnp.sum(jnp.where(mask, jnp.square(difference_from_max), 0.0))
        mse = sum_sq_diff / jnp.maximum(batch_size, 1)

        # # Scaled difference for regret calculation
        # scaled_difference = scale_observation_jax(difference_from_max, state.min_y, state.max_y)

        
        # jax.debug.print("we resetting {}", min_y)
        # # print("initial_actions_mapped", initial_actions_mapped.shape, sam_con, params)
        # obs_raw = compute_y_sampler(initial_actions_mapped[:1, :], sam_con, params)
        # obs_raw = jnp.atleast_1d(obs_raw)

        # # Pad initial observations
        # padded_obs_raw = jnp.full((max_batches, 1), min_y, dtype=jnp.float32)
        # padded_obs_raw = padded_obs_raw.at[:1, 0].set(obs_raw)

        # # Scale and calculate initial metrics
        # scaled_observation = scale_observation_jax(obs_raw, min_y, max_y)
        # current_best_scaled_y = jnp.max(scaled_observation, initial=0.0)
        # avg_scaled_obs = jnp.mean(scaled_observation)

        # # Find best initial X
        # best_idx_in_batch = jnp.argmax(scaled_observation)
        # best_obs_x = initial_actions_mapped[best_idx_in_batch]

        # Calculate initial reward (adjust based on how you want to reward step 0)
        # Using the same formula as step, but improvement/new_best will be 0 implicitly
        initial_reward = (params.r_best * current_best_scaled_y + params.r_obs * avg_scaled_obs) * params.r_scale
        
        # print("initial_reward", padded_obs_raw.shape, initial_actions_normalized.shape, avg_scaled_obs.shape, )

        # print("rsa", obs_raw.shape, initial_actions_normalized.shape, jnp.squeeze(scaled_observation).shape)
        # Initialize state
        last_obs = jnp.reshape(scaled_observation, (max_batches, 1))
        last_raw = jnp.reshape(obs_raw, (max_batches, 1))
        
        state = EnvState(
            tick=0,
            sampler_type_index=sampler_type_index,
            optimum_point=optimum_point,
            min_y=min_y,
            max_y=max_y,
            last_avg_scaled_obs=avg_scaled_obs,
            best_scaled_y_so_far=current_best_scaled_y,
            best_obs_x_so_far=current_best_x,
            batch_size=batch_size,
            achieved_success=False,
            last_action_mapped=initial_actions_mapped,
            last_padded_obs_raw=last_obs,
            last_action=initial_actions_normalized, # Store action taken
            last_raw_obs=last_raw, # Store raw observation
            last_reward=initial_reward,
            done=False, # Reset is not done
            max_steps_in_episode=max_steps, # Set max steps for this episode
            params_for_compute=sam_con, # Pass params for compute_y_sampler
            episode_counter=0, # Initialize episode counter
            key=key # Final key state
        )
        
        

        obs = get_obs(state, params)
        
        
        # jax.debug.print("initial_actions_mapped {}", obs)
        # print("osb", obs["actions"].shape, obs["observations"].shape, obs["reward"].shape, obs["mask"].shape, obs["step"].shape)
        return obs, state
    
    @staticmethod
    def reset_env_keep(
        key: chex.PRNGKey, params: EnvParams, action_dim: int, max_batches: int, state: EnvState
    ) -> Tuple[chex.ArrayTree, EnvState]:
        """Resets the environment to an initial state."""
        key, key_sampler, key_batch, key_action = jax.random.split(key, 4)

      

      
        # Generate initial actions
        initial_actions_normalized = jax.lax.cond(
            params.use_random_action_on_reset,
            lambda k: jax.random.uniform(k, (max_batches, action_dim), minval=-1.0, maxval=1.0),
            lambda k: jnp.zeros((max_batches, action_dim)),
            key_action,
        )
        initial_actions_mapped = map_to_bounds_jax(initial_actions_normalized, params.x_range)
  
        # obs_raw = compute_y_sampler(initial_actions_mapped, state.params_for_compute, params)
        obs_raw = compute_y_sampler_dispatch(initial_actions_mapped, state.params_for_compute, params)
        obs_raw = jnp.atleast_1d(obs_raw)
        
        mask = jnp.arange(max_batches) < state.batch_size
        
        # print("obs_raw", obs_raw.shape, action_mapped.shape, state.params_for_compute, params)

        # # Pad observations
        # # Use jnp.nan or state.min_y for padding? Using min_y as in original.
        # padded_obs_raw = jnp.full((max_batches, 1), state.min_y, dtype=jnp.float32)
        
        # padded_obs_raw = obs_raw.at[state.batch_size:].set(jnp.zeros_like(obs_raw[batch_size:])) # i thought this was fine since the output will always be the same shape
        # # padded_obs_raw = padded_obs_raw.at[:batch_size, 0].set(obs_raw) # this was the original code you provided which gave similar issues with tracing

        # --- Reward Calculation ---
        # Scale the valid observations
        scaled_observation = scale_observation_jax(obs_raw, state.min_y, state.max_y) # Shape: (batch_size,)

        # Calculate metrics
        current_best_scaled_y = jnp.max(jnp.where(mask, scaled_observation, -jnp.inf)) # Handle empty case
        sum_scaled_obs = jnp.sum(jnp.where(mask, scaled_observation, 0.0))
        avg_scaled_obs = sum_scaled_obs / state.batch_size # Average over valid observations

        # Update best_obs_x for the episode
        best_idx_in_batch = jnp.argmax(jnp.where(mask, scaled_observation, -jnp.inf))
        current_best_x = initial_actions_mapped[best_idx_in_batch] # X corresponding to best Y *in this batch*

        # # Improvement metrics
        # avg_improvement = avg_scaled_obs - state.last_avg_scaled_obs
        # new_best_bonus = jnp.maximum(0.0, current_best_scaled_y - state.best_scaled_y_so_far)

        # MSE from maximum (optional penalty)
        difference_from_max = state.max_y - obs_raw
        sum_sq_diff = jnp.sum(jnp.where(mask, jnp.square(difference_from_max), 0.0))
        mse = sum_sq_diff / jnp.maximum(state.batch_size, 1)

        # # Scaled difference for regret calculation
        # scaled_difference = scale_observation_jax(difference_from_max, state.min_y, state.max_y)

        
        # jax.debug.print("we resetting {}", min_y)
        # # print("initial_actions_mapped", initial_actions_mapped.shape, sam_con, params)
        # obs_raw = compute_y_sampler(initial_actions_mapped[:1, :], sam_con, params)
        # obs_raw = jnp.atleast_1d(obs_raw)

        # # Pad initial observations
        # padded_obs_raw = jnp.full((max_batches, 1), min_y, dtype=jnp.float32)
        # padded_obs_raw = padded_obs_raw.at[:1, 0].set(obs_raw)

        # # Scale and calculate initial metrics
        # scaled_observation = scale_observation_jax(obs_raw, min_y, max_y)
        # current_best_scaled_y = jnp.max(scaled_observation, initial=0.0)
        # avg_scaled_obs = jnp.mean(scaled_observation)

        # # Find best initial X
        # best_idx_in_batch = jnp.argmax(scaled_observation)
        # best_obs_x = initial_actions_mapped[best_idx_in_batch]

        # Calculate initial reward (adjust based on how you want to reward step 0)
        # Using the same formula as step, but improvement/new_best will be 0 implicitly
        initial_reward = (params.r_best * current_best_scaled_y + params.r_obs * avg_scaled_obs) * params.r_scale
        
        # print("initial_reward", padded_obs_raw.shape, initial_actions_normalized.shape, avg_scaled_obs.shape, )

        # print("rsa", obs_raw.shape, initial_actions_normalized.shape, jnp.squeeze(scaled_observation).shape)
        # Initialize state
        last_obs = jnp.reshape(scaled_observation, (max_batches, 1))
        
        
        state = state.replace( # Use replace for immutability with dataclasses/Pytrees
            tick=0,
            # sampler_type_index, optimum_point, min_y, max_y, batch_size, params_for_compute remain same
            last_avg_scaled_obs=avg_scaled_obs,
            best_scaled_y_so_far=current_best_scaled_y,
            best_obs_x_so_far=current_best_x,
            achieved_success=False,
            last_action_mapped=initial_actions_mapped, # Store full mapped actions
            last_padded_obs_raw=last_obs, # Store padded obs)
            last_action=initial_actions_normalized, # Store action taken
            last_reward=initial_reward,
            done=False, # Store done flag reflecting current step's outcome
            # max_steps_in_episode remains same
            episode_counter=state.episode_counter + 1, # Initialize episode counter
            key=key # Pass updated key if state carries it
        )
        

        obs = get_obs(state, params)
        
        # jax.debug.print("initial_actions_mapped {}", obs)
        # print("osb", obs["actions"].shape, obs["observations"].shape, obs["reward"].shape, obs["mask"].shape, obs["step"].shape)
        return obs, state

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        """Defines the observation space."""
        
        
        return spaces.Dict(
            {
                "actions": spaces.Box(
                    low=params.x_range[0],
                    high=params.x_range[1],
                    shape=(params.max_batches, params.action_dim),
                  
                ),
                "observations": spaces.Box(
                    low=-jnp.inf, high=jnp.inf,
                    shape=(params.max_batches, 1),
                   
                ),
                "reward": spaces.Box(
                    low=-jnp.inf, high=jnp.inf,
                    shape=(1,),
                  
                ),
                "mask": spaces.Box( # Mask is now implicit via batch_size in state, but can expose if needed
                    low=0, high=params.max_batches,
                    shape=(1,),
                 
                ),
                "step": spaces.Box(
                    low=0, high=params.max_steps_in_episode,
                    shape=(1,),
                  
                )
            }
        )
        
        # return spaces.Dict(
        #     {
        #         "actions": spaces.Box(
        #             low=params.x_range[0],
        #             high=params.x_range[1],
        #             shape=(params.max_batches, params.action_dim),
        #             dtype=jnp.float32,
        #         ),
        #         "observations": spaces.Box(
        #             low=-jnp.inf, high=jnp.inf,
        #             shape=(params.max_batches, 1),
        #             dtype=jnp.float32
        #         ),
        #         "reward": spaces.Box(
        #             low=-jnp.inf, high=jnp.inf,
        #             shape=(1,),
        #             dtype=jnp.float32
        #         ),
        #         "mask": spaces.Box( # Mask is now implicit via batch_size in state, but can expose if needed
        #             low=0, high=params.max_batches,
        #             shape=(1,),
        #             dtype=jnp.int32
        #         ),
        #         "step": spaces.Box(
        #             low=0, high=params.max_steps_in_episode,
        #             shape=(1,),
        #             dtype=jnp.int32
        #         )
        #     }
        # )
        
        
    

    def action_space(self, params: EnvParams) -> spaces.Box:
        """Defines the action space."""
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(params.max_batches, params.action_dim),
        )

# --- Helper JAX functions (must be defined outside the class) ---

def map_to_bounds_jax(actions: chex.Array, x_range: Tuple[float, float]) -> chex.Array:
    """Maps actions from [-1, 1] to the environment's x_range using JAX."""
    
    out_of_bounds_mask = (actions < -1) | (actions > 1)


    
    lower, upper = x_range
    scale = (upper - lower) / 2.0
    shift = (upper + lower) / 2.0
    return scale * actions + shift

def scale_observation_jax(obs: chex.Array, min_y: float, max_y: float) -> chex.Array:
    """Scales observation y to be roughly in [0, 1] using JAX."""
    denominator = max_y - min_y
    # Avoid division by zero/very small numbers
    safe_denominator = jnp.maximum(denominator, 1e-9)
    scaled = (obs - min_y) / safe_denominator
    # Clip to ensure bounds, handles cases where obs might be slightly outside [min_y, max_y]
    return jnp.clip(scaled, 0.0, 1.0)

def get_obs(state: EnvState, params: EnvParams) -> chex.ArrayTree:
    """Constructs the observation dictionary from the current state."""
    
    
    # return {
    #     "actions": state.last_action_mapped.astype(jnp.float64),
    #     "observations": state.last_padded_obs_raw.astype(jnp.float32),
    #     "reward": jnp.array([state.last_reward], dtype=jnp.float32),
    #     "mask": jnp.array([state.batch_size], dtype=jnp.int32), # Expose batch size if needed by agent
    #     "step": jnp.array([state.tick], dtype=jnp.int32)
    # }
    return {
        "actions": state.last_action_mapped,
        "observations": state.last_padded_obs_raw,
        "reward": jnp.array([state.last_reward]),
        "mask": jnp.array([state.batch_size]), # Expose batch size if needed by agent
        "step": jnp.array([state.tick])
    }
    
    

def get_info(state: EnvState, params: EnvParams, done: bool, scaled_difference: chex.Array) -> Dict:
    """Constructs the info dictionary, populated only when done."""
    # Info dict is populated only when the episode is done (Gymnax standard)
    # Otherwise, return an empty dict.
    def true_fn():
        # Calculate final metrics based on the *completed* episode state
        distance = jnp.linalg.norm(state.best_obs_x_so_far - state.optimum_point)
        info_dict = {
            # "final_observation": state.last_padded_obs_raw,
            "episode_length": state.tick,
            # "reward_per_episode": state.total_reward, # Need to accumulate reward in state if desired
            # "batch_mse": state.accumulated_mse / state.tick, # Need accumulation
            "last_avg_scaled_obs": state.last_avg_scaled_obs,
            "best_rewards": state.best_scaled_y_so_far,
            "scaled_diff": jnp.mean(scaled_difference), # Regret from last step only
            "last_scaled_diff": scaled_difference, # Regret from last step only
            # "actions": state.all_actions, # Requires storing history, usually not done
            "success": state.achieved_success,
            "max_x": state.optimum_point,
            "distance_from_max": distance,
        }
        return info_dict

    def false_fn():
        placeholder_dict = {
            # "final_observation": jnp.zeros_like(state.last_padded_obs_raw),
            "episode_length": jnp.array(-1),
            "last_avg_scaled_obs": jnp.array(0.0),
            "best_rewards": jnp.array(-jnp.inf), # Or 0.0 if appropriate
            "scaled_diff": jnp.array(0.0),
            "last_scaled_diff": jnp.zeros_like(scaled_difference),
            "success": jnp.array(False, dtype=bool),
            "max_x": jnp.zeros_like(state.optimum_point),
            "distance_from_max": jnp.array(0.0),
        }
        return placeholder_dict

    return jax.lax.cond(done, true_fn, false_fn)

# ==========================================
# JAX Sampler Implementations (Crucial Part)
# ==========================================
# You MUST rewrite your AckleySampler, CosineSampler, etc., here
# using only JAX primitives.

