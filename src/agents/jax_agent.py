# File: ppo_jax.py (replaces basic_ppo.py and parts of root_agent.py)
import jax
import jax.numpy as jnp
import optax
import rlax
import flax.linen as nn
from flax.training.train_state import TrainState
from functools import partial
import chex # Use chex.dataclass if available, otherwise flax.struct
import flax.struct as struct
from typing import Callable,Tuple, Optional, Any
import jax.experimental.checkify as checkify # Make sure checkify is imported


# Assume ActorCriticModel, model_fns, sampling_impl are imported correctly
# Assume MultiFunctionGymnax Env definition (EnvParams, EnvState) is available
# from .jax_env import MultiFunctionGymnax, EnvParams, EnvState # Example
# from src.models.actor_critic import ActorCriticModel # Example
from src.tasks.envs.jax_env_f.jax_env import MultiFunctionGymnax,  EnvParams, EnvState # Example
# from src.tasks.envs.jax_env.env_params import EnvParams # Example
from src.models.actor_critic import ActorCriticModel # Example
# from src.agents.ppo_dic_inherits.inh_agents.full_params_agent import FullParamsSampling # Example sampling_impl

# Define TrainState for the Agent
class AgentTrainState(TrainState):
    # Add target parameters or other state if needed (e.g., for TD3/SAC)
    pass

@struct.dataclass
class RolloutData:
    """Data collected during rollout"""
    observations: chex.ArrayTree # Shape: (num_envs, rollout_len + 1, *) - includes final obs
    actions: chex.Array # Shape: (num_envs, rollout_len, *)
    rewards: chex.Array # Shape: (num_envs, rollout_len) - reward received at step t+1
    dones: chex.Array # Shape: (num_envs, rollout_len) - done flag at step t+1
    log_probs: chex.Array # Shape: (num_envs, rollout_len, *)
    values: chex.Array # Shape: (num_envs, rollout_len + 1) - includes final value
    actor_preds: chex.Array # Shape: (num_envs, rollout_len, *) - logits/params from actor
    # Include masks if needed by the loss function
    hidden_states: chex.ArrayTree # Shape: (num_envs, rollout_len, *h_dims) - h_0 to h_{T-1}
    # Termination flag d_t *before* taking action a_t
    start_dones: chex.Array # Shape: (num_envs, rollout_len) - d_0 to d_{T-1}
    success: chex.Array # Shape: (num_envs, rollout_len) - success flag at step t+1
    regret: chex.Array # Shape: (num_envs, rollout_len) - regret at step t+1
    best_actions: chex.Array # Shape: (num_envs, rollout_len, *)
    last_step: chex.Array # Shape: (num_envs, rollout_len) - last step before stepping
    
    masks: Optional[chex.Array] = None # Shape: (num_envs, rollout_len, *)
    
    


class PPOAgentJax:
    def __init__(self,
                 env_params: EnvParams, # Pass EnvParams directly
                 env_params_test: EnvParams, # Pass EnvState directly if needed
                 repr_model_fn: Callable,
                 seq_model_fn: tuple[Callable, Callable],
                 actor_fn: Callable,
                 critic_fn: Callable,
                 optimizer: optax.GradientTransformation,
                 sampling_impl_class, # e.g., FullParamsSampling
                 # PPO Hyperparameters
                 rollout_len: int = 128,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 num_minibatches: int = 4,
                 update_epochs: int = 4,
                 norm_adv: bool = True,
                 clip_coef: float = 0.1,
                 ent_coef_schedule: optax.Schedule = lambda _: 0.01, # Example schedule
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: Optional[float] = None,
                 # Add other relevant config...
                 ):

        self.env_params = env_params
        self.env_params_test = env_params_test # Store test params if needed
        self.rollout_len = rollout_len
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.ent_coef_schedule = ent_coef_schedule # Use the schedule directly
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.optimizer = optimizer

        self.sampling_impl = sampling_impl_class(
            action_dim=env_params.action_dim,
            max_batch=env_params.max_batches,
            # Add other necessary args for sampling_impl init
        )
        
        # self.env_params["max_batches"] = jnp.max(jnp.array(env_params["batches"]))
        # Note: Sampling impl no longer needs self.agent reference in pure JAX

        self.seq_fn, self.seq_init = seq_model_fn

        # --- Build Actor-Critic Model ---
        # IMPORTANT: Remove the nn.vmap wrapper here if using jax.vmap for env parallelism
        # The model should operate on single environment inputs (B=1 or just T, ...)
        self.ac_model = ActorCriticModel(repr_model_fn, self.seq_fn, actor_fn, critic_fn)

        # Bind the apply function for convenience
        self.ac_apply = self.ac_model.apply

        # Define a pure function for actor-critic forward pass
        # This will be called inside the rollout scan
        def _actor_critic_step(params, random_key, obs, done, h_prev):
            # Ensure inputs have correct shape for the model (e.g., add time dim if needed)
            # Model expects T=1, potentially B=1 depending on design
            # Assuming model handles obs dict and done shape (1,) and h_prev pytree
            # Add batch dim B=1 before passing to model
            # obs_b = jax.tree_map(lambda x: jnp.expand_dims(x, 0), obs)
            # done_b = jnp.expand_dims(done, 0) # Shape (1,) -> (1, 1) or just (1,)
            # h_prev_b = jax.tree_map(lambda x: jnp.expand_dims(x, 0), h_prev)
            
            obs_b = jax.tree_map(lambda x: jnp.expand_dims(x, 0), obs)
            done_b = jnp.expand_dims(done, 0)
            h_prev_b = jax.tree_map(lambda x: x, h_prev)

            # Add rngs={'random': key} if model uses randomness internally beyond sampling
            act_logits, value, h_next_b = self.ac_apply(
                {'params': params}, obs_b, done_b, h_prev_b # Removed rngs assuming model is deterministic given inputs
            )

            # Remove batch dim B=1
            # print("we are in actor critic step", act_logits.shape, value.shape, h_next_b[0][0].shape)
            
            # h_next = jax.tree_map(lambda x: x.squeeze(0), h_next_b)
            act_logits = act_logits.squeeze(0)
            value = value.squeeze(0)

            return act_logits, value, h_next_b

        self._actor_critic_step = _actor_critic_step # Store for use in rollout

    def init(self, key: chex.PRNGKey) -> AgentTrainState:
        """Initializes agent parameters and optimizer state."""
        key, params_key, seq_key = jax.random.split(key, 3)

        obs_space = MultiFunctionGymnax().observation_space(self.env_params) # Instantiate temporarily

        # --- MODIFICATION HERE ---
        # Access the underlying dictionary of spaces (.spaces)
        # and apply tree_map to that dictionary.
        if hasattr(obs_space, 'spaces') and isinstance(obs_space.spaces, dict):
             spaces_dict = obs_space.spaces
        else:
             # Handle cases where obs_space might not be Dict or structure differs
             # If obs_space itself *is* the dictionary, use it directly.
             # This depends on the exact return type of observation_space()
             # Assuming it returns a spaces.Dict object for this fix:
             raise TypeError(f"Expected obs_space to have a .spaces attribute containing a dict, but got {type(obs_space)}")

        dummy_obs = jax.tree_map(lambda space: jnp.zeros(space.shape, space.dtype), spaces_dict)
        # --- END MODIFICATION ---

        # Add Batch=1, Time=1 dimensions if model expects them
        dummy_obs_b = jax.tree_map(lambda x: jnp.expand_dims(x, 0), dummy_obs)
        dummy_done = jnp.zeros((1,), dtype=bool)
        dummy_h = self.seq_init() # Get initial hidden state structure
        dummy_h_b = jax.tree_map(lambda x: x, dummy_h) # Add Batch=1
        
        
        #This is where the ac is initialsed
        params = self.ac_model.init(
            {'params': params_key}, # Add rngs if needed
            dummy_obs_b, dummy_done, dummy_h_b
        )['params']

        optimizer_state = self.optimizer.init(params)

        # Assuming AgentTrainState is correctly defined
        # from flax.training.train_state import TrainState
        # class AgentTrainState(TrainState): pass # Example definition

        return AgentTrainState.create(
            apply_fn=self.ac_apply,
            params=params,
            tx=self.optimizer  # Pass the optimizer itself
            # REMOVE opt_state=optimizer_state
        )

    # --- Rollout ---
    # @partial(jax.jit, static_argnums=(0,)) # Jit the rollout function
    
    # @partial(jax.jit, static_argnums=(0,))
    @partial(jax.jit, static_argnames=('self', 'max_batches', 'action_dim')) #using the upper one makes it fail later whent the parameters are used
    def rollout(self,
                keys: chex.PRNGKey, # Shape: (num_envs,) - keys for each env
                params: chex.ArrayTree,
                initial_h_states: chex.ArrayTree, # Shape: (num_envs, *)
                initial_obs: chex.ArrayTree, # Shape: (num_envs, *)
                initial_env_states: chex.ArrayTree, # Shape: (num_envs, *)
                env_params: EnvParams, # Static
                max_batches: int,
                action_dim: int,
                ) -> tuple[tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree], RolloutData]:
        """Runs parallel rollouts using vmap and scan."""

     
      
        # In PPOAgentJax class, inside rollout method:

        # Function for a single environment step within the scan
        def _env_step_scan(carry, key_t):
            h_prev, obs, env_state, last_done = carry # last_done is from the PREVIOUS step
            key_model, key_sample, key_env_step, key_env_reset = jax.random.split(key_t, 4) # Need extra key for potential reset

            # 1. Get action from policy
            # Reset hidden state *before* processing obs if the *previous* step was done
            h_reset = self.seq_init() # Get the structure of initial hidden state
            # Select h_prev if last_done is False, h_reset if last_done is True
            h_processed = jax.tree_map(lambda reset_h, prev_h: jax.lax.select(last_done, reset_h, prev_h), h_reset, h_prev)

            #Here the forward pass of the network is done in order to get the networks output
            act_logits, value, h_next = self._actor_critic_step(params, key_model, obs, last_done, h_processed) # Pass last_done for potential model use
            mask = obs.get("mask", None)
            
            #here the network output is used to sample an action but without the normalising flow yet
            #This self.sampling_impl is of type SamplingImplBaseJax
            action = self.sampling_impl.sampling_differ(act_logits, key_sample, mask)
            log_prob = self.sampling_impl.gaussian_log_prob(jnp.expand_dims(action, axis=0), jnp.expand_dims(act_logits, axis=0))

            # 2. Step the environment
            # This step uses the *current* env_state
            current_obs, current_env_state, reward, done, info = MultiFunctionGymnax.step_env(
                key_env_step, env_state, action, env_params, max_batches, action_dim
            )
            
            
            
            def reset_same_fn(_):
                # jax.debug.print("resetting same")
                return MultiFunctionGymnax.reset_env_keep(
                    key_env_reset, env_params, action_dim, max_batches, current_env_state
                )

            def reset_new_fn(_):
                # jax.debug.print("resetting new")
                return MultiFunctionGymnax.reset_env(
                    key_env_reset, env_params, action_dim, max_batches
                )

            def no_reset_fn(_):
                # jax.debug.print("no reset")
                return current_obs, current_env_state

        
            reset_mode = jnp.array(1)  # Default: keep current

            # Criterion A: Success
            # was_successful = info.get("success", jnp.array(False))
            # reset_mode = jnp.where(was_successful, 2, reset_mode)

            # Criterion B: Every N episodes
            reset_frequency = 5
            reset_mode = jnp.where(env_state.episode_counter >= reset_frequency - 1, 2, reset_mode)

            # # Criterion C: Random
            # key_reset_decision, _ = jax.random.split(key_env_reset)
            # reset_probability = 0.2
            # random_reset = jax.random.uniform(key_reset_decision) < reset_probability
            # reset_mode = jnp.where(random_reset, 2, reset_mode)

            # Optional: If you want to allow for reset to the *same* function rather than random new,
            # you could assign mode 1 for that and distinguish how `reset_env` behaves.

            # Final mode decision: only reset if `done` is True
            final_mode = jax.lax.select(done, reset_mode, 0)  # 0 = no reset
         

            next_obs_carry, next_env_state_carry = jax.lax.switch(
                final_mode,
                [no_reset_fn, reset_same_fn, reset_new_fn],
                operand=None
            )
            
            
        
            # 4. Prepare output data for *this* step (before potential reset)
            step_data = {
                "obs": obs, # s_t
                "action": action, # a_t
                "reward": reward, # r_{t+1}
                "done": done, # d_{t+1} - Indicates if THIS step ended the episode
                "value": value, # V(s_t)
                "log_prob": log_prob, # logp(a_t|s_t)
                "actor_pred": act_logits, # logits for a_t
                "mask": mask, # mask for s_t
                "next_obs": current_obs, # s_{t+1} (before reset)
                "hidden_state": h_processed, # h_t (potentially reset based on d_t)
                "start_dones": last_done, # d_t (done signal *before* this step)
                # "step": env_state.tick + 1, # Careful with tick if env resets internally
                "success": info["success"], # Success flag from env step
                "regret": info["scaled_diff"], # Regret from env step
                "best_rewards": info["best_rewards"], # Best actions taken at step t+1
                "last_yes": info["episode_length"], # Last step before stepping
            }
            # Prepare carry for the *next* iteration
            # Pass the done flag from *this* step to be the 'last_done' for the next iteration
            next_carry = (h_next, next_obs_carry, next_env_state_carry, done)
            return next_carry, step_data
        
        # In PPOAgentJax class, inside rollout method:

        def _rollout_single_env(key_env, h_init, obs_init, env_state_init):
            # Initial carry state for the scan
            # The first step always assumes the previous step was *not* done.
            initial_carry = (h_init, obs_init, env_state_init, jnp.array(False, dtype=bool))
            keys_t = jax.random.split(key_env, self.rollout_len) # Keys for each time step

            # Scan over the rollout length (remains the same)
            (final_h, final_obs, final_env_state, final_done), step_data_sequence = jax.lax.scan(
                _env_step_scan, initial_carry, keys_t, length=self.rollout_len
            )

            # Get the value prediction for the final observation s_T
            # Use the 'final_done' flag from the *last* step's transition
            key_final_model, _ = jax.random.split(key_env)
            # Reset final hidden state if the last step resulted in done=True
            h_reset_final = self.seq_init()
            final_h_processed = jax.tree_map(lambda reset_h, prev_h: jax.lax.select(final_done, reset_h, prev_h), h_reset_final, final_h)

            _, final_value, _ = self._actor_critic_step(params, key_final_model, final_obs, final_done, final_h_processed)

            # --- Populate RolloutData (Using corrected obs logic from previous answer) ---
            all_obs = jax.tree_map(
                lambda init, seq_next: jnp.concatenate([jnp.expand_dims(init, 0), seq_next], axis=0),
                obs_init, # s_0
                step_data_sequence["next_obs"] # s_1, ..., s_T (before potential reset)
            ) # Shape (rollout_len + 1, *obs_dims)

            all_values = jnp.concatenate([step_data_sequence["value"], jnp.expand_dims(final_value,0)], axis=0) # V(s_0)..V(s_T)

            rollout = RolloutData(
                observations=all_obs, # s_0 to s_T
                actions=step_data_sequence["action"], # a_0 to a_{T-1}
                rewards=step_data_sequence["reward"], # r_1 to r_T
                dones=step_data_sequence["done"],     # d_1 to d_T (True when episode actually ends)
                log_probs=step_data_sequence["log_prob"],
                values=all_values, # V(s_0) to V(s_T)
                actor_preds=step_data_sequence["actor_pred"],
                masks=step_data_sequence["mask"],
                hidden_states=step_data_sequence["hidden_state"], # h_0 to h_{T-1} (already potentially reset)
                start_dones=step_data_sequence["start_dones"], # d_0 to d_{T-1} (indicates resets)
                success=step_data_sequence["success"], # success flag at step t+1
                regret=step_data_sequence["regret"], # regret at step t+1
                best_actions=step_data_sequence["best_rewards"], # actions taken at step t+1
                last_step=step_data_sequence["last_yes"], # last step before stepping
            )

            # Return the final state *after* the scan (might be a reset state)
            return (final_h, final_obs, final_env_state), rollout

      

        # Vmap over the batch of environments
        vmapped_rollout = jax.vmap(_rollout_single_env, in_axes=(0, 0, 0, 0))
        (final_h, final_obs, final_env_state), trajectory_data = vmapped_rollout(
            keys, initial_h_states, initial_obs, initial_env_states
        )

        return (final_h, final_obs, final_env_state), trajectory_data


    # --- PPO Update ---
    @partial(jax.jit, static_argnums=(0,)) # Jit the update function
    def update(self,
               key: chex.PRNGKey,
               agent_state: AgentTrainState,
               rollout_data: RolloutData,
               update_step: int # Pass current update step for schedules
               ) -> tuple[AgentTrainState, dict]:
        """Performs the PPO update step."""

        # Extract data (shapes are (num_envs, rollout_len / +1, ...))
        obs_T = rollout_data.observations # s_0 to s_T
        actions_T = rollout_data.actions # a_0 to a_{T-1}
        rewards_T = rollout_data.rewards # r_1 to r_T
        dones_T = rollout_data.dones # d_1 to d_T
        log_probs_T = rollout_data.log_probs # logpi(a_t|s_t)
        values_T = rollout_data.values # V(s_0) to V(s_T)
        actor_preds_T = rollout_data.actor_preds # Actor net output for a_0..a_{T-1}
        masks_T = rollout_data.masks # Masks for a_0..a_{T-1}
        
        hidden_states_T = rollout_data.hidden_states # h_0 to h_{T-1}
        start_dones_T = rollout_data.start_dones # d_0 to d_{T-1}

       
        print("why is this not used??!!!!!!", log_probs_T.shape)
        
       
        
        discounts_T = self.gamma * (1.0 - dones_T)

        # Vectorize rlax.lambda_returns, keeping lambda scalar
        Glambda_fn = jax.vmap(rlax.lambda_returns, in_axes=(0, 0, 0, None)) # Corrected in_axes

        # Calculate lambda-returns: G_t^lambda for t = 0..T-1
        # Pass V(s_1)...V(s_T) as the value input
        targets = Glambda_fn(
            rewards_T,        # r_1..r_T (Shape T)
            discounts_T,      # discount_1..discount_T (Shape T)
            values_T[:, 1:],  # V(s_1)..V(s_T) (Shape T) <-- FIX HERE
            self.gae_lambda   # Scalar lambda_
        )
        
        # jax.debug.print("targets {} ", targets)
        
        advantages = targets - values_T[:, :-1]
                
        
       
        num_envs = rewards_T.shape[0]
        batch_size = num_envs * self.rollout_len

       
        rollout_len = actions_T.shape[1] # T

        # We will minibatch over the 'num_envs' dimension
        batch_size = num_envs # Minibatching happens over environments

        # Observations s_0 to s_{T-1} needed for loss calc
        observations_loss = jax.tree_map(lambda x: x[:, :-1], obs_T) # Shape (B, T, *)
        # Initial hidden state h_0 for each sequence
        hidden_states_init = jax.tree_map(lambda x: x[:, 0], hidden_states_T) # Shape (B, *h)
        
        # --- Additions ---
        # hidden_states_flat = flatten_rollout_data(hidden_states_T) # h_0 to h_{T-1}
        # start_dones_flat = flatten_rollout_data(start_dones_T)     # d_0 to d_{T-1}
        # --- End Additions ---
        # TODO: Need to handle recurrent states (hiddens) if using sequence batching

        # Normalize advantages globally
        if self.norm_adv:
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
            
        # Inside PPOAgentJax class

        def _ppo_loss_fn_single(
                            params, random_key,
                            obs_seq,        # Shape (T, *) - s_0..s_{T-1}
                            actions_seq,    # Shape (T, *) - a_0..a_{T-1}
                            logp_old_seq,   # Shape (T, *)
                            adv_seq,        # Shape (T,)
                            targets_seq,    # Shape (T,)
                            masks_seq,      # Shape (T, *)
                            start_dones_seq,# Shape (T,)   - d_0..d_{T-1}
                            h_init          # Shape (*h) - h_0
                            # update_step might be needed here if ent_coef_schedule depends on it
                            ):
            """Calculates PPO loss for a SINGLE sequence."""
            key_model, key_entropy = jax.random.split(random_key)

            # --- 1. Apply model (NO internal vmap needed now) ---
            # Inputs are already single sequence (T, *) and single h_init (*)
            act_logits_seq, values_seq, _ = self.ac_apply(
                {'params': params}, obs_seq, start_dones_seq, h_init
            )
            # Output shapes (T, *)
            
          

            # --- 2. Calculate probs, entropy for the single sequence ---
            # These sampling_impl methods must work on (T, *) inputs
            logp_new_seq = self.sampling_impl.gaussian_log_prob(actions_seq, act_logits_seq) # Shape (T, *)
            
            print("logp_new_seq", logp_new_seq.shape, actions_seq.shape, act_logits_seq.shape)
            # Entropy might return (T,) or scalar mean. Assume (T,) for now.
            entropy_seq = self.sampling_impl.entropy(act_logits_seq, masks_seq, key=key_entropy) # Shape (T,)
            # We need the mean entropy for the loss term
            mean_entropy = entropy_seq.mean() # Mean over T

            # --- 3. Calculate Policy Loss for the sequence ---
            logratio = logp_new_seq - logp_old_seq
            
            logratio = jnp.clip(logratio, -20, 20) # Clip logratio to avoid numerical issues
            ratio = jnp.exp(logratio)
            
            print("logratio", logratio.shape, logp_new_seq.shape, logp_old_seq.shape, ratio.shape, adv_seq.shape)
            
            pg_loss1 = -adv_seq * ratio
            pg_loss2 = -adv_seq * jnp.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            # Mean over T for the policy loss of this sequence
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # --- 4. Calculate Value Loss for the sequence ---
            # Mean over T for the value loss of this sequence
            v_loss = 0.5 * jnp.square(values_seq - targets_seq).mean()

            # --- 5. Total Loss for the sequence ---
            # Need update_step if schedule is used. Pass it into this function.
            # ent_coef = self.ent_coef_schedule(update_step)
            ent_coef = -0.01 # Placeholder - pass update_step if schedule is needed
            total_loss = pg_loss + self.vf_coef * v_loss - ent_coef * mean_entropy

            # --- 6. Metrics for the sequence ---
            approx_kl = jnp.mean((ratio - 1) - logratio) # Mean over T
            
            # print("sag", logp_new_seq.shape, logp_old_seq.shape, adv_seq.shape, targets_seq.shape, masks_seq.shape, start_dones_seq.shape)
            metrics = {
                "loss/loss": total_loss, "loss/loss_policy": pg_loss, "loss/loss_value": v_loss,
                "loss/loss_entropy": mean_entropy * ent_coef, "loss/kl_approx": approx_kl,
                "loss/ratio": jnp.mean(ratio), "loss/log_old": jnp.mean(logp_old_seq), "loss/log_new": jnp.mean(logp_new_seq),
                "loss/values_pred": jnp.mean(values_seq), "loss/target": jnp.mean(targets_seq)# Use actual coef if passed
            }
            # Return scalar loss and metrics dict for this single sequence
            return total_loss, metrics


        def _update_epoch(carry, _):
            agent_state_epoch, key_epoch = carry
            key_epoch, key_perm = jax.random.split(key_epoch)

           # Shuffle environment indices
            perms = jax.random.permutation(key_perm, num_envs)

            # Function to shuffle data along the first (environment) dimension
            def shuffle_env_dim(x):
                if x is None: return None
                return jax.tree_map(lambda leaf: leaf[perms], x)

            observations_shuffled = shuffle_env_dim(observations_loss) # Shape (B, T, *)
            actions_shuffled = shuffle_env_dim(actions_T)               # Shape (B, T, *)
            log_probs_shuffled = shuffle_env_dim(log_probs_T)           # Shape (B, T, *)
            advantages_shuffled = shuffle_env_dim(advantages)           # Shape (B, T)
            targets_shuffled = shuffle_env_dim(targets)                 # Shape (B, T)
            masks_shuffled = shuffle_env_dim(masks_T)                   # Shape (B, T, *)
            start_dones_shuffled = shuffle_env_dim(start_dones_T)       # Shape (B, T)
            hidden_init_shuffled = shuffle_env_dim(hidden_states_init)  # Shape (B, *h)
            


            # Minibatch update loop (Scan over minibatches)
            minibatch_size = num_envs // self.num_minibatches
            
            def _update_minibatch(carry_mb, i):
                agent_state_mb, key_mb = carry_mb
                key_mb, key_loss = jax.random.split(key_mb)
                start = i * minibatch_size
                
                # agent_state_mb, key_mb = carry_mb
                # key_mb, key_loss = jax.random.split(key_mb)
                # start = i * minibatch_size
                # end = start + minibatch_size # Not strictly needed for dynamic_slice

                # Slice function for pytrees along first dimension
                def slice_tree(t):
                    if t is None: return None
                    return jax.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, minibatch_size, axis=0), t)
                # Slice function for arrays along first dimension
                def slice_arr(arr):
                        if arr is None: return None
                        return jax.lax.dynamic_slice_in_dim(arr, start, minibatch_size, axis=0)


                # --- Slice data (as before) to get minibatch shapes (B, T, *) ---
                # ... (slicing code using slice_tree, slice_arr) ...
                mb_obs_seq = slice_tree(observations_shuffled)
                mb_actions_seq = slice_tree(actions_shuffled)
                mb_logp_old_seq = slice_tree(log_probs_shuffled)
                mb_adv_seq = slice_arr(advantages_shuffled)
                mb_targets_seq = slice_arr(targets_shuffled)
                mb_masks_seq = slice_tree(masks_shuffled)
                mb_start_dones_seq = slice_arr(start_dones_shuffled)
                mb_h_init = slice_tree(hidden_init_shuffled)

                # --- Vmap the single-sequence loss function ---
                # Pass update_step if ent_schedule is needed inside _ppo_loss_fn_single
                vmapped_loss_fn = jax.vmap(
                    _ppo_loss_fn_single, # Call the single-sequence method
                    # Map over axis 0 for all data args, keep params/key shared
                    in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0)
                )

                # --- Define function for jax.grad (computes mean loss) ---
                def batch_loss_for_grad(params, key, obs, act, logp, adv, targ, mask, dones, h_init):
                    # Calculate loss & metrics for the whole batch using vmap
                    batch_loss, batch_metrics = vmapped_loss_fn(
                        params, key, obs, act, logp, adv, targ, mask, dones, h_init
                    )
                    # Return scalar mean loss for grad, keep batch metrics as aux
                    return batch_loss.mean(), batch_metrics

                # --- Calculate gradients w.r.t. mean batch loss ---
                grad_target_fn = jax.value_and_grad(batch_loss_for_grad, has_aux=True)
                (mean_loss, batch_metrics), grads = grad_target_fn(
                    agent_state_mb.params, key_loss,
                    mb_obs_seq, mb_actions_seq, mb_logp_old_seq, mb_adv_seq, mb_targets_seq, mb_masks_seq,
                    mb_start_dones_seq, mb_h_init
                )

                # --- Apply Gradients ---
                agent_state_new = agent_state_mb.apply_gradients(grads=grads)

                # --- Aggregate metrics over the minibatch ---
                final_mb_metrics = jax.tree_map(jnp.mean, batch_metrics)

                return (agent_state_new, key_mb), final_mb_metrics # Return aggregated metrics
            
      

            (agent_state_epoch, key_epoch), mb_metrics_all = jax.lax.scan(
                _update_minibatch, (agent_state_epoch, key_epoch), jnp.arange(self.num_minibatches)
            )
            # Aggregate metrics across minibatches
            epoch_metrics = jax.tree_map(jnp.mean, mb_metrics_all)
            return (agent_state_epoch, key_epoch), epoch_metrics

        # Run update epochs
        key, key_epochs = jax.random.split(key)
        (final_agent_state, _), epoch_metrics_all = jax.lax.scan(
            _update_epoch, (agent_state, key_epochs), None, length=self.update_epochs
        )
        # Aggregate metrics across epochs
        update_metrics = jax.tree_map(jnp.mean, epoch_metrics_all)

        # Add grad norm if needed (calculate before applying updates inside loop)
        # Add param norm
        update_metrics["params_l2"] = l2_norm(final_agent_state.params)
        # Add other final metrics

        return final_agent_state, update_metrics
    
    
    # Define the main evaluation function that iterates
    def evaluate(self,
                key: chex.PRNGKey,
                params: chex.ArrayTree,
                env_params_dict: dict, # Now accepts a dictionary
                num_eval_episodes: int,
                action_dim: int, # Still needed if not derivable from env_params_dict values
                max_batches: int, # Still needed if not derivable
                max_steps_in_episode: int, # Still needed if not derivable
                eval_mode: str = 'ppo', # Default to mean evaluation
                ) -> dict:
        """Runs evaluation for multiple environment configurations."""

        all_eval_metrics = {}
        env_names = list(env_params_dict.keys())
        
        print("env_names", params.keys())
        
        
        for i, env_name in enumerate(env_names):
            key, subkey = jax.random.split(key) # Use a new key for each env type
            current_env_params = env_params_dict[env_name]

            # print(f"--- Evaluating on {env_name} ---")
            # Call the JIT-compiled function for this specific env_params
            # Ensure action_dim, max_batches, max_steps are consistent or derived from current_env_params
            # If they vary per env_params, get them from current_env_params inside the loop
            # Example: action_dim = current_env_params.action_dim (if defined)
            
            eval_metrics_single_type = self.evaluate_func_type(
                subkey,
                params,
                current_env_params, # Pass the single EnvParams object
                num_eval_episodes,
                action_dim, # Pass consistent values or derive from current_env_params
                max_batches,
                max_steps_in_episode,
                eval_mode 
            )

            # Prefix metrics with env_name and add to the overall results
            for metric_name, value in eval_metrics_single_type.items():
                all_eval_metrics[f"eval_{env_name}/{metric_name}"] = value

        # Optionally add an overall average across types if meaningful
        # Example: Calculate mean return across all evaluated types
        # all_returns = [v for k, v in all_eval_metrics.items() if k.endswith('/episode_return')]
        # if all_returns:
        #     all_eval_metrics["eval/mean_return_across_types"] = jnp.mean(jnp.array(all_returns))

        return all_eval_metrics

    # --- Evaluation ---
    @partial(jax.jit, static_argnames=('self', 'num_eval_episodes', 'action_dim', 'max_batches', 'max_steps_in_episode', 'eval_mode')) # Jit the evaluation function
    def evaluate_func_type(self,
                 key: chex.PRNGKey, # Single key to split
                 params: chex.ArrayTree,
                 env_params: dict,
                 num_eval_episodes: int,
                 action_dim: int, # Action dimension for the environment
                 max_batches: int, # Number of batches for the environment step
                 max_steps_in_episode: int,
                 eval_mode: str,
                 ) -> dict:
        """Runs evaluation episodes in parallel."""
        print("evaluate_func_type", params.keys())
        # --- Reset environments ---
        keys_reset = jax.random.split(key, num_eval_episodes)
        vmapped_reset = jax.vmap(MultiFunctionGymnax.reset_env, in_axes=(0, None, None, None))
        batch_obs, batch_env_states = vmapped_reset(keys_reset, env_params, action_dim, max_batches)

        # Initialize batch of agent hidden states
        # Use a dummy key for init shape - actual state is per-env
        key, h_init_key = jax.random.split(keys_reset[0]) # Just need one key for shape
        single_h_init = self.seq_init()
        batch_h_states = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), num_eval_episodes, 0), single_h_init)

        # --- Define scan function for episode steps ---
        def _eval_step_scan(carry, _): # Scan over steps, input not used
            h_prev, obs, env_state, done, key_carry = carry
            key_carry, key_model, key_sample, key_env, key_rand_act = jax.random.split(key_carry, 5)
            
            
            def ppo_action_fn():
                # Standard PPO evaluation action selection
                act_logits, _, h_next = self._actor_critic_step(params, key_model, obs, done, h_prev)
                mask = obs.get("mask", None)
                # Use deterministic action (mode/mean) or sampling
                action = self.sampling_impl.sampling_differ(act_logits, key_sample, mask) # Keep sampling or use mode
                return action, h_next

            def random_action_fn():
                # Sample random actions in [-1, 1] range (or env action space if different)
                # Assumes action space is [-1, 1]^action_dim
                # Need shape (max_batches, action_dim) expected by step_env
                random_action_normalized = jax.random.uniform(
                    key_rand_act,
                    shape=(max_batches, action_dim), # Ensure correct shape
                    minval=-1.0, maxval=1.0
                )
                # No model step, hidden state just passes through or resets if done
                h_reset = self.seq_init()
                h_next = jax.tree_map(lambda reset_h, prev_h: jax.lax.select(done, reset_h, prev_h), h_reset, h_prev)
                return random_action_normalized, h_next
            
            action, h_next = jax.lax.cond(
                eval_mode == "eval_random",
                random_action_fn,
                ppo_action_fn
            )



            # Step environment - only if not already done
            def step_fn():
                 return MultiFunctionGymnax.step_env(key_env, env_state, action, env_params, max_batches, action_dim)
            def no_step_fn():
                 # Return current state and 0 reward/info etc. if already done
                 # Ensure types match step_env output
                 place_holder = {
                    # "final_observation": jnp.zeros((max_batches, 1)), # Placeholder for final observation
                    "episode_length": 0,
                    # "reward_per_episode": state.total_reward, # Need to accumulate reward in state if desired
                    # "batch_mse": state.accumulated_mse / state.tick, # Need accumulation
                    "last_avg_scaled_obs": jnp.zeros(()),
                    "best_rewards":jnp.zeros(()),
                    "scaled_diff": jnp.zeros(()), # Regret from last step only
                    "last_scaled_diff": jnp.zeros(()), # Regret from last step only
                    # "actions": state.all_actions, # Requires storing history, usually not done
                    "success": jnp.array(False),
                    "max_x": jnp.zeros((action_dim,)),
                    "distance_from_max": jnp.zeros(()),
                }
                 
                 return obs, env_state, jnp.array(0.0), jnp.array(True), place_holder#{} # Env dependent

            next_obs, next_env_state, reward, next_done, info = jax.lax.cond(
                done, no_step_fn, step_fn
            )

            # Accumulate reward, update done state
            reward_accum = jax.lax.select(done, 0.0, reward) # Don't add reward if already done
            new_done = done | next_done

            next_carry = (h_next, next_obs, next_env_state, new_done, key_carry)
            # Output per-step data: reward, info (if needed)
            step_output = {"reward": reward_accum, "info": info, "done": new_done}
            return next_carry, step_output

        # --- Vmap the scan over episodes ---
        @partial(jax.vmap, in_axes=(0, 0, 0, 0)) # Vmap over key, h, obs, env_state
        def _run_single_episode(key_episode, h_init, obs_init, env_state_init):
            initial_carry = (h_init, obs_init, env_state_init, jnp.array(False), key_episode)
            # Scan for max episode steps
            _, step_outputs = jax.lax.scan(
                _eval_step_scan, initial_carry, None, length=max_steps_in_episode
            )
            # step_outputs: pytree with leaves of shape (max_steps, *)

            # Calculate episode statistics from step_outputs
            total_reward = jnp.sum(step_outputs["reward"])
            # Find actual episode length (first step where done is true)
            true_done_idx = jnp.argmax(step_outputs["done"])
            # If done never happens, length is max_steps. Add 1 because index is 0-based.
            episode_length = jnp.where(jnp.any(step_outputs["done"]), true_done_idx + 1, max_steps_in_episode)
            
            success = step_outputs["info"]["success"][-1]

            # Calculate mean for 'regret'
            regret = step_outputs["info"]["scaled_diff"][-1]
            # Calculate mean for 'rewards'
            best_action =step_outputs["info"]["best_rewards"][-1]
            
            

            

            # Extract final info (assuming info dict is structured correctly by env)
            # This requires careful handling based on how info is populated in MultiFunctionGymnax.step_env
            # Example: get success rate if present in the info of the final step
            # final_info = jax.tree_map(lambda x: x[episode_length-1], step_outputs["info"]) # Get info from last actual step
            # success_rate = final_info.get("success", 0.0) # Example access

            # Return aggregated results per episode
            return {
                "episode_return": total_reward,
                "episode_length": episode_length,
                "best_action": best_action,
                "regret": regret,
                "success": success,
                # "eval/success_rate": success_rate, # Add other metrics from info
            }

        # Run vmapped evaluation
        keys_eval = jax.random.split(key, num_eval_episodes)
        eval_results_per_episode = _run_single_episode(keys_eval, batch_h_states, batch_obs, batch_env_states)

        # Average results across episodes
        eval_metrics_mean = jax.tree_map(jnp.mean, eval_results_per_episode)

        # TODO: Add evaluation metrics from env 'info' dictionary if needed
        # This requires the env's step_env to put JAX-compatible values into info,
        # and the _eval_step_scan and aggregation logic needs to handle them.

        return eval_metrics_mean
    


# --- Helper Functions ---
def l2_norm(pytree):
    """Computes the L2 norm of a pytree of arrays."""
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(pytree)]))

