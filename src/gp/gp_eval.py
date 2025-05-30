from src.gp.gpjax import BayesianOptimizer
import jax
import jax.numpy as jnp
import numpy as np
import logging
from typing import Dict, Any, Callable
import chex
from src.tasks.envs.jax_env_f.jax_env import MultiFunctionGymnax
# from src.tasks.envs.jax_env_f.jax_function_samplers import compute_y_sampler
# from src.tasks.envs.jax_env_f.jax_function_samplers import compute_y_sampler_dispatch

import gpjax as gpx
from tqdm import tqdm


SUPPORTED_GPJAX_KERNELS: Dict[str, Callable[..., gpx.kernels.AbstractKernel]] = {
    "matern52": gpx.kernels.Matern52,
    "matern32": gpx.kernels.Matern32,
    "RBF": gpx.kernels.RBF,
    "polynomial": gpx.kernels.Polynomial,
    "linear": gpx.kernels.Linear,
    "periodic": gpx.kernels.Periodic,
    "white": gpx.kernels.White,
    "arc_cosine": gpx.kernels.ArcCosine,
    "matern12": gpx.kernels.Matern12,
    "exponential": gpx.kernels.PoweredExponential,
    "eigen_comp": gpx.kernels.EigenKernelComputation,
    "rational_quadratic": gpx.kernels.RationalQuadratic,
    "RFF": gpx.kernels.RFF,
    
}

def run_bo_evaluation(
                          key: chex.PRNGKey,
                          test_environments: dict, # Dict of EnvParams
                          num_eval_episodes: int,
                          # Add other BO specific args: bo_restarts, kappa, initial_samples etc.
                          bo_restarts: int = 10,
                          bo_initial_random_samples: int = 0,
                          ) -> dict:
        """Runs evaluation using the Bayesian Optimization agent."""

        # Import your BO class

        all_bo_metrics = {}

        for env_name, env_params in test_environments.items():
            # print(f"Evaluating {env_name} with Bayesian Optimization")
            key, env_key = jax.random.split(key)
            env_keys = jax.random.split(env_key, num_eval_episodes)

            episode_returns = []
            episode_lengths = []
            episode_successes = []
            episode_final_regrets = []
            epsiode_best_actions = []
            

            # Loop over episodes for this env type
            for episode_idx in tqdm(range(num_eval_episodes)):
                # print(f"Evaluating {env_name} - Episode {episode_idx+1}/{num_eval_episodes}")
                episode_key = env_keys[episode_idx]
                ep_key, bo_key, reset_key = jax.random.split(episode_key, 3)

                # 1. Initialize BO Agent for the episode
                # Ensure bounds are jax arrays for BO class
                
                obs_init_eval, current_env_state = MultiFunctionGymnax.reset_env(
                        reset_key, env_params, env_params.action_dim, env_params.max_batches
                    )
                
                bounds = (jnp.full(env_params.action_dim, current_env_state.params_for_compute['common']['bounds'][0]),
                          jnp.full(env_params.action_dim,current_env_state.params_for_compute['common']['bounds'][1]))
                
                kernel_type = env_name.split('_kernel')[0]
                
                kern_func = SUPPORTED_GPJAX_KERNELS.get(kernel_type, gpx.kernels.Matern52)
                
                bo_optimizer = BayesianOptimizer(
                    search_space_bounds=bounds,
                    # kernel=gpx.kernels.Matern52(active_dims=list(range(env_params.action_dim))),
                    kernel=kern_func(active_dims=list(range(env_params.action_dim))),
                    key=bo_key
                    
                    # Add other BO params: kernel, kappa etc. if configurable
                )
                
                initial_X = current_env_state.last_action_mapped
                initial_Y = current_env_state.last_raw_obs
                
                
                bo_optimizer.update(initial_X, initial_Y)

                # # 2. Initial Random Samples (Optional but Recommended for BO)
                # if bo_initial_random_samples > 0:
                #     ep_key, init_sample_key = jax.random.split(ep_key)
                    
                #     # Evaluate these initial points using the env
                #     # We need to reset the env to get the *correct* function instance
                #     # This implies the env needs to be reset *per BO episode* if function varies
                #     # Let's reset once here to get the function for this BO run
                #     obs_init_eval, current_env_state = MultiFunctionGymnax.reset_env(
                #         reset_key, env_params, env_params.action_dim, env_params.max_batches
                #     )
                    
                #     initial_X = jax.random.uniform(
                #         init_sample_key,
                #         shape=(bo_initial_random_samples, env_params.action_dim),
                #         minval=current_env_state.params_for_compute['common']['bounds'][0],
                #         maxval=current_env_state.params_for_compute['common']['bounds'][1],
                #     )
                    
                #     print(f"Initial random samples: {initial_X}", current_env_state.params_for_compute['common']['bounds'])
                    
                #     # Evaluate function at initial points
                #     # Ensure compute_y_sampler handles batch evaluation
                #     initial_Y = compute_y_sampler(initial_X, current_env_state.params_for_compute, env_params)
                #     initial_Y = jnp.reshape(initial_Y, (-1, 1)) # Ensure shape (n, 1)
                #     # Update BO with initial samples
                #     bo_optimizer.update(initial_X, initial_Y)


                # 3. BO Episode Loop (Standard Python Loop)
                # We need to use the 'current_env_state' from the reset above if the function is fixed per BO run
                # Or reset the env at the start of the step-loop if function can change?
                # Let's assume function is fixed for the duration of one BO evaluation episode.
                # Need to get the *initial observation* from the reset above.
                # obs = obs_init_eval # Use the observation returned by reset_env
                # The 'env_state' holds the function params for compute_y_sampler

                ep_return = 0.0
                ep_done = False
                ep_step = 0
                final_info = {} # Store final info dict

                for step in range(current_env_state.max_steps_in_episode):
                    # print(f"Episode {episode_idx+1}/{num_eval_episodes}, Step {step+1}/{env_params.max_steps_in_episode}")
                    if ep_done: break # Stop if already done

                    ep_key, step_key, suggest_key = jax.random.split(ep_key, 3)
                    bo_optimizer.state = bo_optimizer.state.replace(key=suggest_key) # Update key for suggest

                    # Get suggestion from BO
                    action = bo_optimizer.suggest(n_restarts=bo_restarts, batch_size=current_env_state.batch_size)
                    
                    squashed_action = 2 * (action - bounds[0]) / (bounds[1] - bounds[0]) - 1
                    

                    # Step the JAX environment (use state with correct function params)
                    # Need to pass state, action, env_params. Key may not be needed by step_env.
                    # Use np.array() if step_env expects numpy action
                    obs, current_env_state, reward, done, info = MultiFunctionGymnax.step_env(
                        step_key, # Pass key if step_env uses it
                        current_env_state,
                        squashed_action, # Pass JAX array if step_env expects it
                        env_params,
                        env_params.max_batches, # Pass necessary static args
                        env_params.action_dim
                    )
                    reward_val = float(np.array(reward)) # Convert reward to scalar float
                    ep_done = bool(np.array(done))

                    # Update BO model with the new observation (action, reward)
                    # BO typically optimizes f(x), so reward IS the observation y
                    # jax.debug.print("BO update with action: {} and reward: {}", reward_val, ep_done)
                    
                    
                    bo_optimizer.update(current_env_state.last_action_mapped, current_env_state.last_raw_obs)

                    ep_return += reward_val
                    ep_step = step + 1
                    if ep_done:
                        final_info = info # Store info from the step where done became true
                        
                del bo_optimizer
                
                jax.clear_caches()

                # End of episode loop
                episode_returns.append(ep_return)
                episode_lengths.append(ep_step)
                # Extract metrics from final_info
                success = final_info.get("success", False)
                scaled_diff = final_info.get("scaled_diff", np.nan)
                episode_successes.append(float(success)) # Convert bool to float
                episode_final_regrets.append(float(scaled_diff))
                best_action = final_info.get("best_rewards", np.nan)
                epsiode_best_actions.append(best_action)
                

            # Average metrics for this environment type
            all_bo_metrics[f"eval_{env_name}/episode_return"] = np.mean(episode_returns)
            all_bo_metrics[f"eval_{env_name}/episode_length"] = np.mean(episode_lengths)
            all_bo_metrics[f"eval_{env_name}/success"] = np.mean(episode_successes)
            all_bo_metrics[f"eval_{env_name}/regret"] = np.nanmean(episode_final_regrets) # Use nanmean
            all_bo_metrics[f"eval_{env_name}/best_action"] = np.mean(epsiode_best_actions) # Use nanmean
            

        return all_bo_metrics