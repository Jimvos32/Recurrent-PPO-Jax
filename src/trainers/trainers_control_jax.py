import json
import numpy as np
import optax
import pandas as pd
import rlax
import wandb
import gymnasium as gym
import time
import logging
from argparse import Namespace
from src.trainers.base_trainer import BaseTrainer
from collections import OrderedDict
from src.tasks.envs.minigrid_env import create_minigrid_env_onehot,create_minigrid_env_pixel, create_sampling_env, create_multi_dim_env, create_multi_batch_env, create_mbatch, create_multi_dim, create_multi_fun_env, create_multi_batch_env, create_multi_dim, create_jax_env
from src.agents.a2c import A2CAgent
from src.agents.ppo import PPOAgent
from src.agents.ppo_vae import PPOAgentVAE
from src.agents.ppo_norm import PPOAgentNorm
from src.model_fns import *
from src.tasks.envs.wrappers import *
from src.trainers.utils import *
from gymnasium.wrappers import AutoResetWrapper
from omegaconf import DictConfig, OmegaConf
from src.model_fns.norm_fns import planar_flow, autoregressive_flow
from src.agents.ppo_dic_inherits.basic_ppo import BasePPO
from src.agents.ppo_dic_inherits.vae_ppo import VAEPPO
from src.agents.ppo_dic_inherits.inh_agents.full_params_agent import FullParamsSampling
from src.agents.ppo_dic_inherits.inh_agents.standard_sampling import SamplingImplBase
from src.agents.ppo_dic_inherits.inh_agents.cor_gmm_agent import CorrelatedGaussianMixture 
from src.agents.ppo_dic_inherits.inh_agents.low_mvn_agent import LowRankMVN 
from src.model_fns.repr_fns import dict_unpack_model, dict_unpack_mask
from src.agents.ppo_dic_inherits.inh_agents.flow_jax_agent import FlowMVN




# File: trainers_control_jax_refactored.py (rename or replace trainers_control_jax.py)
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
import logging
import wandb # Keep wandb for logging outside JAX
from functools import partial

# Assume PPOAgentJax, EnvParams, create_env_params are imported
from src.tasks.envs.jax_env_f.jax_env import MultiFunctionGymnax, EnvState
from src.agents.ppo_dic_inherits.inh_agents.jax_full_params import FullParamsSamplingJax
from src.agents.ppo_dic_inherits.inh_agents.flow_jax import FlowMVNJax
from src.agents.jax_agent import PPOAgentJax
from src.tasks.envs.jax_env_f.jax_function_samplers import create_env_params, EnvParams, initialize_sampler
# from .jax_env import EnvParams # Example
# from .env_utils import create_env_params # Example

# Assume model_fns, etc. are imported
# from src.model_fns import *

logger = logging.getLogger(__name__)

# Remove or adapt get_env_initializers - no longer needed for Gym wrappers
# The trainer will work directly with EnvParams and the JAX env's pure functions

class ControlTrainerJaxRefactored: # Renamed class

    def __init__(self, trainer_config, env_config, global_args, key, wandb_run):
        """
        Initialize the JAX-based PPO trainer.
        """
        self.trainer_config = trainer_config
        self.env_config = env_config # Expects dict/OmegaConf container
        self.global_config = global_args
        self.wandb_run = wandb_run
        self.key = key
        self.key, self.eval_key = jax.random.split(self.key)

        self.num_envs = self.trainer_config['num_envs']
        self.rollout_len = self.trainer_config['rollout_len']
        self.num_updates = self.global_config['steps'] // (self.num_envs * self.rollout_len)

        # --- Environment Setup ---
        # Create EnvParams using the helper function
        conf = OmegaConf.to_container(env_config, resolve=True)
        bathes = jnp.array(conf.get("batches", [1,1]))
        m_batch = jnp.max(bathes)
        b_train, b_test = jnp.split(bathes, 2)
        
        conf["max_batches"] = m_batch
        conf["batches"] = b_train
        conf["function_types"] = conf.get("env_train", ["pol"])
        self.env_params_train = create_env_params(conf)
        
        conf["batches"] = b_test
        self.test_environments = {}
        test_functions = conf.get("env_test", ["pol"])
        
        for func in test_functions:
            f_conf = conf.copy()
            f_conf["function_types"] = [func]
            self.test_environments[func] = create_env_params(f_conf)
            
        # self.env_params_train = create_env_params(conf)
        # conf["batches"] = b_test
        # conf["function_types"] = conf.get("env_test", ["pol"])
        
        
        
        # self.env_params_test = create_env_params(conf)

        # --- Agent Setup ---
        # Get model functions (adapt this part based on your config structure)
        if self.trainer_config.seq_model.name == 'lstm':
            seq_model_fn = seq_model_lstm(**self.trainer_config['seq_model'])
        # Add other seq models (gru, gtrxl)
        else:
            raise ValueError(f"Unknown seq_model: {self.trainer_config.seq_model.name}")

        # Define actor/critic heads (adapt based on your config)
        # repr_fn = dict_unpack_model() # Assuming simple unpack for JAX env obs dict
        repr_fn = dict_unpack_mask(batch_expand_hidden=env_config["batch_expand_hidden"], 
                                 batch_combine_hidden=env_config["batch_combine_hidden"], 
                                 step_expand_hidden=env_config["step_expand_hidden"], 
                                 input_combine_hidden=env_config["input_combine_hidden"])
        
        critic_fn = critic_model(self.trainer_config['d_critic'])
        
        
        sampling_impl_class, actor_fn = self.select_policy_distribution(self.env_config['pol_dist'], self.env_params_train)
        
        # Optimizer setup
        optimizer_config = dict(self.trainer_config.optimizer)
        lr_config = optimizer_config.pop("learning_rate")
        ent_config = self.trainer_config['ent_coef']
        # stability_config = self.trainer_config['stability_coef'] # If used

        lr_schedule = optax.polynomial_schedule(**lr_config)
        ent_schedule = optax.polynomial_schedule(**ent_config)
        # stability_schedule = optax.polynomial_schedule(**stability_config)

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.trainer_config['max_grad_norm']),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=lr_schedule, # Pass schedule directly if adamw supports it, or update later
                 **optimizer_config
            ),
        )

        # Instantiate the JAX Agent
        self.agent = PPOAgentJax(
            env_params=self.env_params_train,
            env_params_test=self.test_environments,
            repr_model_fn=repr_fn,
            seq_model_fn=seq_model_fn,
            actor_fn=actor_fn,
            critic_fn=critic_fn,
            optimizer=optimizer,
            sampling_impl_class=sampling_impl_class,
            rollout_len=self.rollout_len,
            gamma=self.trainer_config.get('gamma', 0.99),
            gae_lambda=self.trainer_config.get('gae_lambda', 0.95),
            num_minibatches=self.trainer_config.get('num_minibatches', 4),
            update_epochs=self.trainer_config.get('update_epochs', 4),
            norm_adv=self.trainer_config.get('norm_adv', True),
            clip_coef=self.trainer_config.get('clip_coef', 0.1),
            ent_coef_schedule=ent_schedule, # Pass the schedule object
            vf_coef=self.trainer_config.get('vf_coef', 0.5),
            max_grad_norm=self.trainer_config.get('max_grad_norm', 0.5),
            target_kl=self.trainer_config.get('target_kl', None),
        )
        
        self.action_dim = self.env_params_train.action_dim
        self.max_batches = self.env_params_train.max_batches

        # --- Initialize Agent and Environment States ---
        self.key, agent_init_key, env_init_key = jax.random.split(self.key, 3)

        # Initialize agent state (params, optimizer state)
        self.agent_state = self.agent.init(agent_init_key)

        # Initialize environment states (vmap over num_envs)
        env_keys = jax.random.split(env_init_key, self.num_envs)
        vmapped_reset = jax.vmap(MultiFunctionGymnax.reset_env, in_axes=(0, None, None, None))
        self.batch_obs, self.batch_env_states = vmapped_reset(env_keys, self.env_params_train, self.env_params_train.action_dim, self.env_params_train.max_batches)
        
        

        # Initialize batch of agent hidden states
        _, h_init_key = jax.random.split(self.key) # Use any key for shape init
        single_h_init = self.agent.seq_init()
        self.batch_h_states = jax.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, 0), self.num_envs, 0), single_h_init
        )

        # --- Logging/Eval Setup ---
        self.step_count = 0
        self.log_interval = self.global_config.log_interval
        self.eval_interval = self.global_config.get('eval_interval', None)
        self.next_log_step = self.log_interval
        self.next_eval_step = self.eval_interval if self.eval_interval is not None else float('inf')
        self.results_data = [] # For storing metrics for final table

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.num_updates} updates.")
        run_mode = self.trainer_config.get('run_mode', 'train')
        
        interactions = 0
        avg_batch_size = jnp.array(self.env_params_train.batches).mean()
        
        

        for update_idx in range(self.num_updates):
            start_time = time.time()
            step_metrics = {}

            if run_mode == 'train':
                # --- Run one step of interaction and update ---
                self.key, step_key = jax.random.split(self.key)
                step_metrics, (self.batch_h_states, self.batch_obs, self.batch_env_states), self.agent_state = self._train_step(
                    step_key,
                    self.agent_state,
                    self.batch_h_states,
                    self.batch_obs,
                    self.batch_env_states
                )
                
            elif run_mode in ['random', 'bo']:
                pass
            
           
            
            # --- Logging ---
            self.step_count += self.num_envs * self.rollout_len
            end_time = time.time()
            # print("Step metrics:", start_time, end_time) # Debugging line
            
            
            sps = self.num_envs * self.rollout_len / ((end_time - start_time) + 1e-6) # Steps per second)

            if run_mode == "train" and self.step_count >= self.next_log_step:
                self.next_log_step += self.log_interval
                
                actions = np.reshape(step_metrics['actions'], ((self.num_envs * self.rollout_len), step_metrics['actions'].shape[2], step_metrics['actions'].shape[3]))
                histo_dic = {}
                for k in range(actions.shape[1]):
                    histo_dic['env/action dimension ' + str(k)] = wandb.Histogram(actions[:, k]) # Log each action dimension separately
                    
                step_metrics.pop('actions') # Remove actions from metrics to avoid confusion
                
                interactions += self.num_envs * self.rollout_len * avg_batch_size
                
                step_metrics['interactions'] = interactions
                
                
                log_metrics = {
                    "step": self.step_count,
                    "update": update_idx,
                    "sps": sps,
                    **step_metrics # Add metrics from agent update
                }
                # Log to wandb (convert JAX arrays to NumPy/Python scalars)
                log_metrics_np = jax.tree_map(lambda x: np.array(x).item() if np.isscalar(x) else np.array(x), log_metrics)
                log_metrics_np = log_metrics_np | histo_dic # Add histogram data to log metrics
                
                
                if self.wandb_run:
                    self.wandb_run.log(log_metrics_np)
                logger.info(f"Update: {update_idx}, Step: {self.step_count}, SPS: {sps:.2f}, Loss: {log_metrics_np['loss/loss']:.4f}")
                self.results_data.append(log_metrics_np) # Store for final summary

            # --- Evaluation ---
            if self.step_count >= self.next_eval_step:
                self.next_eval_step += self.eval_interval
                logger.info(f"Evaluating at step {self.step_count}...")
                self.key, eval_key = jax.random.split(self.key)
                
                current_eval_mode = "ppo" # Default if training
                if run_mode == "eval_random":
                    current_eval_mode = "random"
                elif run_mode == "eval_bo":
                    # --- BO Evaluation Handling ---
                    # Option A: Integrate into agent.evaluate (requires agent modifications)
                    # current_eval_mode = "bo"
                    # Option B: Call a separate BO evaluation function (Recommended)
                    logger.info("BO Evaluation - Skipping agent.evaluate, assuming separate process.")
                    eval_metrics = {} # No metrics from agent.evaluate for BO here
                    # Trigger your separate BO evaluation script/function if needed
                    # run_bo_evaluation(...)
                    # --- End BO Evaluation Handling ---
                else: # Default PPO evaluation during training or if mode is just 'train'
                     current_eval_mode = "ppo"
                     
                if current_eval_mode in ["ppo", "random"]:
                
                    eval_metrics = self.agent.evaluate(
                        self.eval_key,
                        self.agent_state.params,
                        self.test_environments,
                        self.global_config['eval_episodes'],
                        self.env_params_train.action_dim,
                        self.env_params_train.max_batches,
                        self.env_params_train.max_steps_in_episode,
                        eval_mode=current_eval_mode,
                    )
                    # Log eval metrics
                    eval_metrics_np = jax.tree_map(lambda x: np.array(x).item() if np.isscalar(x) else np.array(x), eval_metrics)
                    eval_metrics_np['step'] = self.step_count # Add step count
                    # logger.info(f"Evaluation Results: {eval_metrics_np}")
                    if self.wandb_run:
                        self.wandb_run.log(eval_metrics_np)
                    # Optionally add eval metrics to results_data too

            # --- Checkpointing (Add logic if needed) ---
            # if self.checkpoint_dir and update_idx % self.checkpoint_interval == 0:
            #     save_checkpoint(self.agent_state, self.checkpoint_dir, update_idx)

        logger.info("Training finished.")


    # --- JITted Training Step ---
    @partial(jax.jit, static_argnums=(0,)) # Jit the combined rollout and update step
    def _train_step(self, key, agent_state, h_states, obs, env_states):
        """Performs one rollout and update cycle."""
        key_rollout, key_update = jax.random.split(key)
        rollout_keys = jax.random.split(key_rollout, self.num_envs)
        
        # print("Rollout keys shape:", obs["actions"].shape) # Debugging line

        # Rollout
        (final_h, final_obs, final_env_states), trajectory_data = self.agent.rollout(
            rollout_keys, agent_state.params, h_states, obs, env_states, self.env_params_train, 
            max_batches=self.env_params_train.max_batches, action_dim=self.env_params_train.action_dim
        )
        
        # for k in obs.keys():
        #     print(k, obs[k].shape)
        
        

        # Update
        # Pass current update step count if needed by schedules inside agent.update
        current_update_step = agent_state.step # Assuming TrainState tracks steps
        final_agent_state, update_metrics = self.agent.update(
            key_update, agent_state, trajectory_data, current_update_step
        )
        
        
       
            
            
        # print("Update metrics:", trajectory_data.success.keys()) # Debugging line
        
        # print("Trajectory data keys:", trajectory_data.success)
        
      
        # best_action = trajectory_data.best_action
        
        valid_mask = (trajectory_data.last_step != -1)

        # 2. Calculate means using the 'where' argument
        # jnp.mean(array, where=mask, initial=0.0) calculates the mean of 'array'
        # ONLY where 'mask' is True. If the mask is all False, the sum is 'initial' (0.0),
        # and the count is 0, resulting in NaN by default unless initial is carefully chosen
        # or you handle the count=0 case separately if needed (but often NaN or 0.0 is acceptable).
        # For averaging purposes, letting it potentially return NaN if no elements match is standard.
        # If you prefer 0.0 instead of NaN when no elements match, check the count first or
        # use jnp.nansum/jnp.sum and divide manually with protection.
        # But the direct mean is usually simplest.

        # Calculate mean for 'success'
        # Boolean True/False will be treated as 1.0/0.0
        mean_valid_success = jnp.mean(trajectory_data.success.astype(jnp.float32), where=valid_mask) # Cast bool to float

        # Calculate mean for 'regret'
        mean_valid_regret = jnp.mean(trajectory_data.regret, where=valid_mask)

        # Calculate mean for 'rewards'
        mean_valid_reward = jnp.mean(trajectory_data.rewards, where=valid_mask)

        # Calculate mean for 'best_actions'
        mean_valid_best_action = jnp.mean(trajectory_data.best_actions, where=valid_mask)


        # Add metrics (consider adding the count of valid steps too)
        
        
        # any_sic = trajectory_data.success.any()
        # a = trajectory_data.rewards > 0.0
        # any_reward = trajectory_data.rewards.any()
        
        # jax.debug.print("Success: {} Regret: {} Reward: {} Best Action: {}", mean_valid_success, mean_valid_regret, any_reward, any_sic) # Debugging line

        # Check if any element in the mask is True
        # any_out_of_bounds = out_of_bounds_mask.any()
        
        
        update_metrics['env/success_rate'] = mean_valid_success
        update_metrics['env/regret'] = mean_valid_regret
        update_metrics['env/mean_reward'] = mean_valid_reward
        update_metrics['env/best_action_mean'] = mean_valid_best_action # Renamed slightly for clarity
        update_metrics['env/mean_valid_success'] = mean_valid_success # Optional: Store mean valid success separately
        
        update_metrics['actions'] = trajectory_data.actions # Optional: Store mean valid regret separately
        # print("Actions shape:", trajectory_data.actions.shape) # Debugging line
        
        # jax.debug.print("Mean valid success: {} {}", valid_mask, trajectory_data.last_step) # Debugging line

        # Optional: Add the count of valid steps for context
        num_valid_steps = jnp.sum(valid_mask)
        update_metrics['num_valid_steps'] = num_valid_steps
        

        # Return metrics and updated states
        return update_metrics, (final_h, final_obs, final_env_states), final_agent_state

    def get_summary_table(self):
        # Convert results_data list of dicts to pandas DataFrame or similar
        if not self.results_data:
            return "{}" # Empty json
        try:
            import pandas as pd
            df = pd.DataFrame(self.results_data)
            return df.to_json(orient='records', default_handler=str)
        except ImportError:
            import json
            return json.dumps(self.results_data, default=str)
        
        
    def select_policy_distribution(self, policy, env_params):
        if policy == 'flow_jax_mvn':
            
            sampling_imp = FlowMVNJax
            
            policy_out = env_params.action_dim
            # policy_out = eval_env.unwrapped.action_dim * eval_env.unwrapped.max_batches
            # policy_out = covariance + covariance * (covariance + 1) // 2
            
            

            actor_fn = mvn_flow_head(policy_out, shared_seq_sizes=self.trainer_config['d_actor'], 
                                        policy_hidden_sizes=self.trainer_config['actor_params_hidden'])
            return sampling_imp, actor_fn
            
        elif policy == 'full_params':
            sampling_impl_class = FullParamsSamplingJax # Choose based on config
            
            output_size = env_params.max_batches * env_params.action_dim
            actor_fn = standard_action_head(output_size, seq_hidden_sizes=self.trainer_config['d_actor'], 
                                                policy_layers=self.trainer_config['actor_params_hidden'])
            
            return sampling_impl_class, actor_fn
            

    
