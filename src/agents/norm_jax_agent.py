# File: ppo_jax.py (Modified for Flow Chain Integration)
import jax
import jax.numpy as jnp
import optax
import rlax
import flax.linen as nn
from flax.training.train_state import TrainState
from functools import partial
import chex
import flax.struct as struct
from typing import Callable, Tuple, Optional, Any, Type, List, Dict # Add List, Dict
import equinox as eqx

# --- Flowjax Imports ---
from flowjax.bijections import AbstractBijection, Chain # Import Chain
# Example layer types:
# from flowjax.bijections import BlockAutoregressiveNetwork, Planar, LeakyTanh 

# Assume ActorCriticModel, model_fns, sampling_impl are imported correctly
from src.tasks.envs.jax_env_f.jax_env import MultiFunctionGymnax, EnvParams, EnvState
from src.models.actor_critic import ActorCriticModel
# Assume FlowMVNJax is defined as in the previous response (ID: flowjax_updated_sampler_v2)
# from .flow_jax import FlowMVNJax 

from src.agents.normalising_flow.flow_construction import create_flow_chain
from flowjax.bijections import Tanh



# Define TrainState for the Agent
class AgentTrainState(TrainState):
    pass

@struct.dataclass
class RolloutData:
    """Data collected during rollout"""
    # ... (fields remain the same) ...
    observations: chex.ArrayTree 
    actions: chex.Array 
    rewards: chex.Array 
    dones: chex.Array 
    log_probs: chex.Array 
    values: chex.Array 
    actor_preds: chex.Array 
    hidden_states: chex.ArrayTree 
    start_dones: chex.Array 
    success: chex.Array 
    regret: chex.Array 
    best_actions: chex.Array 
    last_step: chex.Array 
    masks: Optional[chex.Array] = None 


class NormPPOAgentJax:
    def __init__(self,
                 env_params: EnvParams,
                 env_params_test: EnvParams,
                 repr_model_fn: Callable,
                 seq_model_fn: tuple[Callable, Callable],
                 actor_fn: Callable,
                 critic_fn: Callable,
                 optimizer: optax.GradientTransformation,
                 sampling_impl_class, # e.g., FlowMVNJax
                 # --- Flow Configuration (List of Layers) ---
                 # List of tuples: [(LayerClass1, layer_kwargs1), (LayerClass2, layer_kwargs2), ...]
                 flow_layer_configs: List[Tuple[Type[AbstractBijection], Dict[str, Any]]] | None = None,
                 # --- PPO Hyperparameters ---
                 rollout_len: int = 128,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 num_minibatches: int = 4,
                 update_epochs: int = 4,
                 norm_adv: bool = True,
                 clip_coef: float = 0.1,
                 ent_coef_schedule: optax.Schedule = lambda _: 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: Optional[float] = None,
                 # Add other relevant config...
                 ):

        self.env_params = env_params
        self.env_params_test = env_params_test
        self.rollout_len = rollout_len
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.ent_coef_schedule = ent_coef_schedule
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        # ... (rest of hyperparameter initializations) ...
        self.optimizer = optimizer
        
        self.action_dim = env_params.action_dim 

        # --- Sampling Implementation (e.g., FlowMVNJax) ---
        self.sampling_impl = sampling_impl_class(
            action_dim=self.action_dim,
            max_batch=env_params.max_batches,
        )
        

        self.seq_fn, self.seq_init = seq_model_fn

        # --- Build Actor-Critic Model (Flax) ---
        self.ac_model = ActorCriticModel(repr_model_fn, self.seq_fn, actor_fn, critic_fn)
        self.ac_apply = self.ac_model.apply 

        # --- Flow Module Configuration ---
        self.flow_layer_configs = flow_layer_configs if flow_layer_configs else []
        self.static_flow_structure: Optional[AbstractBijection] = None # Initialize attribute
        
        
        init_key_for_structure_only = jax.random.PRNGKey(0)
        temp_flow_obj = create_flow_chain( 
            key=init_key_for_structure_only, # Key here might not matter if layers are deterministic in structure
            action_dim=self.action_dim, 
            layer_configs=self.flow_layer_configs
        )
        if temp_flow_obj is not None:
            _, self.static_flow_structure = eqx.partition(temp_flow_obj, eqx.is_array)
            print(f"AGENT INIT: Stored static_flow_structure with shape: {self.static_flow_structure.shape}")
        else:
            print("AGENT INIT: No flow configured, static_flow_structure is None.")
 
        # Flow chain instance will be created and stored in parameters during init

        # --- Define Actor-Critic Step Helper ---
        # (No changes needed from previous version)
        def _actor_critic_step(ac_params, random_key, obs, done, h_prev):
            obs_b = jax.tree_map(lambda x: jnp.expand_dims(x, 0), obs)
            done_b = jnp.expand_dims(done, 0)
            h_prev_b = jax.tree_map(lambda x: x, h_prev) 
            act_logits_b, value_b, h_next_b = self.ac_apply({'params': ac_params}, obs_b, done_b, h_prev_b)
            act_logits = act_logits_b.squeeze(0)
            value = value_b.squeeze(0)
            return act_logits, value, h_next_b 
        self._actor_critic_step = _actor_critic_step

    def init(self, key: chex.PRNGKey) -> AgentTrainState:
        """Initializes agent parameters (AC and Flow Chain) and optimizer state."""
        key, ac_key, flow_init_key = jax.random.split(key, 3)

        # --- Initialize Actor-Critic (Flax) ---
        # (Same as before)
        obs_space = MultiFunctionGymnax().observation_space(self.env_params)
        if hasattr(obs_space, 'spaces') and isinstance(obs_space.spaces, dict):
             spaces_dict = obs_space.spaces
        else:
             raise TypeError(f"Expected obs_space structure not found.")
        dummy_obs = jax.tree_map(lambda space: jnp.zeros(space.shape, space.dtype), spaces_dict)
        dummy_obs_b = jax.tree_map(lambda x: jnp.expand_dims(x, 0), dummy_obs)
        dummy_done = jnp.zeros((1,), dtype=bool)
        dummy_h = self.seq_init()
        dummy_h_b = jax.tree_map(lambda x: x, dummy_h) 
        ac_params = self.ac_model.init({'params': ac_key}, dummy_obs_b, dummy_done, dummy_h_b)['params']
        
        dynamic_params_for_state = {'ac': ac_params}

        if self.static_flow_structure is not None: # Check if flow is configured
            # Create the initial flow object with actual parameters using flow_init_key
            actual_flow_obj = create_flow_chain( 
                key=flow_init_key, 
                action_dim=int(self.action_dim.item()) if isinstance(self.action_dim, jnp.ndarray) else int(self.action_dim), # Ensure Python int
                layer_configs=self.flow_layer_configs
            )
            if actual_flow_obj is not None:
                dynamic_flow_parts, _ = eqx.partition(actual_flow_obj, eqx.is_array)
                dynamic_params_for_state['flow_dynamic'] = dynamic_flow_parts
            else: # Should not happen if self.static_flow_structure is not None
                dynamic_params_for_state['flow_dynamic'] = {} 
        else:
            dynamic_params_for_state['flow_dynamic'] = {} # Placeholder if no flow
            
            

        # # --- Combine Parameters ---
        # combined_params = {'ac': ac_params}
        # if flow_chain_obj is not None:
        #     # Store the single Chain object which is a PyTree containing all layer params
        #     combined_params['flow'] = flow_chain_obj 

        # --- Initialize Optimizer State ---
        optimizer_state = self.optimizer.init(dynamic_params_for_state)

        # --- Create Train State ---
        return AgentTrainState.create(
            apply_fn=self.ac_apply, 
            params=dynamic_params_for_state, # Store combined parameters
            tx=self.optimizer
        )

    # --- Rollout ---
    # (No changes needed from the previous version )
    # It correctly extracts params['flow'] (which is now the Chain object)
    # and passes it to the sampling_impl methods.
    def rollout(self, keys: chex.PRNGKey, dynamic_params: chex.ArrayTree, initial_h_states: 
        chex.ArrayTree, initial_obs: chex.ArrayTree, initial_env_states: chex.ArrayTree, 
        env_params: EnvParams, max_batches: int, action_dim: int) -> tuple[tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree], RolloutData]:
        # ... (code is identical to the version in jax_agent_with_flow) ...
        ac_params_rollout = dynamic_params['ac']
        dynamic_flow_parts_rollout = dynamic_params.get('flow_dynamic', {})

        def _env_step_scan(carry, key_t):
            h_prev, obs, env_state, last_done = carry 
            key_model, key_sample, key_env_step, key_env_reset = jax.random.split(key_t, 4)
            
            current_flow_obj_rollout = None
            if self.static_flow_structure is not None: # Use agent's static structure
                current_flow_obj_rollout = eqx.combine(dynamic_flow_parts_rollout, self.static_flow_structure)
                
            h_reset = self.seq_init() 
            h_processed = jax.tree_map(lambda reset_h, prev_h: jax.lax.select(last_done, reset_h, prev_h), h_reset, h_prev)
            act_logits, value, h_next = self._actor_critic_step(ac_params_rollout, key_model, obs, last_done, h_processed) 
            mask = obs.get("mask", None)
            # Pass the Chain object here
            action = self.sampling_impl.sampling_differ(act_logits, current_flow_obj_rollout, key_sample, mask)
            log_prob = self.sampling_impl.gaussian_log_prob(jnp.expand_dims(action, axis=0), jnp.expand_dims(act_logits, axis=0), current_flow_obj_rollout)
            # log_prob = log_prob.squeeze(0) 
            # ... (rest of _env_step_scan logic) ...
            current_obs, current_env_state, reward, done, info = MultiFunctionGymnax.step_env(key_env_step, env_state, action, env_params, max_batches, action_dim)
            def reset_same_fn(_): return MultiFunctionGymnax.reset_env_keep(key_env_reset, env_params, action_dim, max_batches, current_env_state)
            def reset_new_fn(_): return MultiFunctionGymnax.reset_env(key_env_reset, env_params, action_dim, max_batches)
            def no_reset_fn(_): return current_obs, current_env_state
            reset_mode = jnp.array(1); reset_frequency = 5
            reset_mode = jnp.where(env_state.episode_counter >= reset_frequency - 1, 2, reset_mode)
            final_mode = jax.lax.select(done, reset_mode, 0)
            next_obs_carry, next_env_state_carry = jax.lax.switch(final_mode, [no_reset_fn, reset_same_fn, reset_new_fn], operand=None)
            step_data = {"obs": obs, "action": action, "reward": reward, "done": done, "value": value, "log_prob": log_prob, "actor_pred": act_logits, "mask": mask, "next_obs": current_obs, "hidden_state": h_processed, "start_dones": last_done, "success": info["success"], "regret": info["scaled_diff"], "best_rewards": info["best_rewards"], "last_yes": info["episode_length"],}
            next_carry = (h_next, next_obs_carry, next_env_state_carry, done)
            return next_carry, step_data

        def _rollout_single_env(key_env, h_init, obs_init, env_state_init):
            initial_carry = (h_init, obs_init, env_state_init, jnp.array(False, dtype=bool))
            keys_t = jax.random.split(key_env, self.rollout_len) 
            (final_h, final_obs, final_env_state, final_done), step_data_sequence = jax.lax.scan(_env_step_scan, initial_carry, keys_t, length=self.rollout_len)
            key_final_model, _ = jax.random.split(key_env)
            h_reset_final = self.seq_init()
            final_h_processed = jax.tree_map(lambda reset_h, prev_h: jax.lax.select(final_done, reset_h, prev_h), h_reset_final, final_h)
            _, final_value, _ = self._actor_critic_step(ac_params_rollout, key_final_model, final_obs, final_done, final_h_processed)
            all_obs = jax.tree_map(lambda init, seq_next: jnp.concatenate([jnp.expand_dims(init, 0), seq_next], axis=0), obs_init, step_data_sequence["next_obs"])
            all_values = jnp.concatenate([step_data_sequence["value"], jnp.expand_dims(final_value,0)], axis=0)
            rollout = RolloutData(observations=all_obs, actions=step_data_sequence["action"], rewards=step_data_sequence["reward"], dones=step_data_sequence["done"], log_probs=step_data_sequence["log_prob"], values=all_values, actor_preds=step_data_sequence["actor_pred"], masks=step_data_sequence["mask"], hidden_states=step_data_sequence["hidden_state"], start_dones=step_data_sequence["start_dones"], success=step_data_sequence["success"], regret=step_data_sequence["regret"], best_actions=step_data_sequence["best_rewards"], last_step=step_data_sequence["last_yes"],)
            return (final_h, final_obs, final_env_state), rollout

        vmapped_rollout = jax.vmap(_rollout_single_env, in_axes=(0, 0, 0, 0))
        (final_h, final_obs, final_env_state), trajectory_data = vmapped_rollout(keys, initial_h_states, initial_obs, initial_env_states)
        return (final_h, final_obs, final_env_state), trajectory_data


    # --- PPO Update ---
    # (No changes needed from the previous version )
    # It correctly extracts params['flow'] (the Chain object) in _ppo_loss_fn_single
    # and passes it to the sampling_impl methods. Gradients flow through the Chain.
    @partial(jax.jit, static_argnums=(0,)) 
    def update(self, key: chex.PRNGKey, agent_state: AgentTrainState, rollout_data: RolloutData, update_step: int) -> tuple[AgentTrainState, dict]:
        # ... (code is identical to the version in jax_agent_with_flow) ...
        dynamic_params_current = agent_state.params
        
        
        
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
        # Observations s_0 to s_{T-1} needed for loss calc
        observations_loss = jax.tree_map(lambda x: x[:, :-1], obs_T) # Shape (B, T, *)
        # Initial hidden state h_0 for each sequence
        hidden_states_init = jax.tree_map(lambda x: x[:, 0], hidden_states_T) # Shape (B, *h)
        
        actions_T = rollout_data.actions; log_probs_T = rollout_data.log_probs; masks_T = rollout_data.masks
        if self.norm_adv: advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def _ppo_loss_fn_single(params_for_loss_calc: chex.ArrayTree, random_key, obs_seq, actions_seq, logp_old_seq, adv_seq, targets_seq, masks_seq, start_dones_seq, h_init):
            key_model, key_entropy = jax.random.split(random_key)
            ac_params_loss = params_for_loss_calc['ac']
        
            reconstructed_flow_obj_loss = None
            if self.static_flow_structure is not None: # From agent instance
                # params_for_loss_calc['flow_dynamic'] has the current dynamic flow params
                # jax.debug.print("flow_dynamic params shape: {} {} {}", params_for_loss_calc['flow_dynamic'][0].params,  params_for_loss_calc['flow_dynamic'][-1].params, ac_params_loss['actor']['MLP_1']['Dense_0'])
                reconstructed_flow_obj_loss = eqx.combine(
                    params_for_loss_calc.get('flow_dynamic', {}), # Default to empty if not present
                    self.static_flow_structure
                )
            
          
            # Recompute AC outputs
            act_logits_seq, values_seq, _ = self.ac_apply({'params': ac_params_loss}, obs_seq, start_dones_seq, h_init)
            # values_seq = values_seq.squeeze(-1) 
            # Recompute log probs and entropy, passing the Chain object
            logp_new_seq = self.sampling_impl.gaussian_log_prob(actions_seq, act_logits_seq, reconstructed_flow_obj_loss) 
            entropy_seq = self.sampling_impl.entropy(act_logits_seq, masks_seq, reconstructed_flow_obj_loss, key=key_entropy) 
            mean_entropy = entropy_seq.mean() 
            # Policy Loss
            logratio = logp_new_seq - logp_old_seq
            ratio = jnp.exp(jnp.clip(logratio, -20, 20))
            pg_loss1 = -adv_seq * ratio; pg_loss2 = -adv_seq * jnp.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
            # Value Loss
            v_loss = 0.5 * jnp.square(values_seq - targets_seq).mean()
            # Total Loss
            ent_coef = self.ent_coef_schedule(update_step)
            
            total_loss = pg_loss + self.vf_coef * v_loss - ent_coef * mean_entropy
            total_loss = pg_loss
            # Metrics
            approx_kl = jnp.mean((ratio - 1) - logratio)
            metrics = {"loss/loss": total_loss, "loss/loss_policy": pg_loss, "loss/loss_value": v_loss, "loss/loss_entropy": mean_entropy * ent_coef, "loss/kl_approx": approx_kl, "loss/ratio": jnp.mean(ratio), "loss/log_old": jnp.mean(logp_old_seq), "loss/log_new": jnp.mean(logp_new_seq), "loss/values_pred": jnp.mean(values_seq), "loss/target": jnp.mean(targets_seq), "loss/entropy_coefficient": ent_coef}
            return total_loss, metrics

        def _update_epoch(carry, _):
            agent_state_epoch, key_epoch = carry
            key_epoch, key_perm = jax.random.split(key_epoch); perms = jax.random.permutation(key_perm, num_envs)
            def shuffle_env_dim(x): return None if x is None else jax.tree_map(lambda leaf: leaf[perms], x)
            observations_shuffled = shuffle_env_dim(observations_loss); actions_shuffled = shuffle_env_dim(actions_T)
            log_probs_shuffled = shuffle_env_dim(log_probs_T); advantages_shuffled = shuffle_env_dim(advantages)
            targets_shuffled = shuffle_env_dim(targets); masks_shuffled = shuffle_env_dim(masks_T)
            start_dones_shuffled = shuffle_env_dim(start_dones_T); hidden_init_shuffled = shuffle_env_dim(hidden_states_init)
            minibatch_size = num_envs // self.num_minibatches
            
            def _update_minibatch(carry_mb, i):
                agent_state_mb, key_mb = carry_mb; key_mb, key_loss = jax.random.split(key_mb); start = i * minibatch_size
                def slice_tree(t): return None if t is None else jax.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, minibatch_size, axis=0), t)
                def slice_arr(arr): return None if arr is None else jax.lax.dynamic_slice_in_dim(arr, start, minibatch_size, axis=0)
                mb_obs_seq = slice_tree(observations_shuffled); mb_actions_seq = slice_tree(actions_shuffled)
                mb_logp_old_seq = slice_tree(log_probs_shuffled); mb_adv_seq = slice_arr(advantages_shuffled)
                mb_targets_seq = slice_arr(targets_shuffled); mb_masks_seq = slice_tree(masks_shuffled)
                mb_start_dones_seq = slice_arr(start_dones_shuffled); mb_h_init = slice_tree(hidden_init_shuffled)
                vmapped_loss_fn = jax.vmap(_ppo_loss_fn_single, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0))
                def batch_loss_for_grad(params, key, obs, act, logp, adv, targ, mask, dones, h_init):
                    batch_loss, batch_metrics = vmapped_loss_fn(params, key, obs, act, logp, adv, targ, mask, dones, h_init)
                    return batch_loss.mean(), batch_metrics
                grad_target_fn = jax.value_and_grad(batch_loss_for_grad, has_aux=True)
                (mean_loss, batch_metrics), grads = grad_target_fn(agent_state_mb.params, key_loss, mb_obs_seq, mb_actions_seq, mb_logp_old_seq, mb_adv_seq, mb_targets_seq, mb_masks_seq, mb_start_dones_seq, mb_h_init)
                agent_state_new = agent_state_mb.apply_gradients(grads=grads) 
                final_mb_metrics = jax.tree_map(jnp.mean, batch_metrics)
                return (agent_state_new, key_mb), final_mb_metrics

            (agent_state_epoch, key_epoch), mb_metrics_all = jax.lax.scan(_update_minibatch, (agent_state_epoch, key_epoch), jnp.arange(self.num_minibatches))
            epoch_metrics = jax.tree_map(jnp.mean, mb_metrics_all)
            return (agent_state_epoch, key_epoch), epoch_metrics

        key, key_epochs = jax.random.split(key)
        (final_agent_state, _), epoch_metrics_all = jax.lax.scan(_update_epoch, (agent_state, key_epochs), None, length=self.update_epochs)
        update_metrics = jax.tree_map(jnp.mean, epoch_metrics_all)
        update_metrics["params_l2"] = l2_norm(final_agent_state.params)
        print("update_metrics", final_agent_state.params.keys())
        
        return final_agent_state, update_metrics


    # --- Evaluation Methods ---
    # (No changes needed from the previous version )
    # Ensure evaluate_func_type extracts params['ac'] and params['flow'] correctly.
    @partial(jax.jit, static_argnames=('self', 'num_eval_episodes', 'action_dim', 'max_batches', 'max_steps_in_episode', 'eval_mode', 'static_flow_structure'))
    def evaluate_func_type(self, key: chex.PRNGKey, dynamic_params: chex.ArrayTree, 
                static_flow_structure: Optional[AbstractBijection], env_params: dict, num_eval_episodes: int, 
                action_dim: int, max_batches: int, max_steps_in_episode: int, eval_mode: str) -> dict:
        # ... (code is identical to the version in jax_agent_with_flow) ...
        # --- Reset environments ---
        ac_params_eval = dynamic_params['ac']
    
        # Reconstruct the full flow object if a static structure was provided
        if static_flow_structure is not None:
            # dynamic_params['flow_dynamic'] contains the JAX array leaves of the original flow object
            flow_params_obj_eval = eqx.combine(dynamic_params['flow_dynamic'], static_flow_structure)
        else:
            flow_params_obj_eval = None # No flow was used
            
            

        
        keys_reset = jax.random.split(key, num_eval_episodes)
        vmapped_reset = jax.vmap(MultiFunctionGymnax.reset_env, in_axes=(0, None, None, None))
        batch_obs, batch_env_states = vmapped_reset(keys_reset, env_params, action_dim, max_batches)
        key, h_init_key = jax.random.split(keys_reset[0]) 
        single_h_init = self.seq_init()
        batch_h_states = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), num_eval_episodes, 0), single_h_init)
        
        # print("outer_callgggg", flow_params_obj_eval.shape)
        # print("flow_params_obj_eval", type(flow_params_obj_eval))
        # jax.debug.print("flow_params_obj_eval shapeggg: {}", flow_params_obj_eval.shape)

        def _eval_step_scan(carry, _): 
            h_prev, obs, env_state, done, key_carry = carry
            key_carry, key_model, key_sample, key_env, key_rand_act = jax.random.split(key_carry, 5)
            def ppo_action_fn():
                act_logits, _, h_next = self._actor_critic_step(ac_params_eval, key_model, obs, done, h_prev)
                mask = obs.get("mask", None)
                # print("outer_call", flow_params_obj_eval.shape)
                # jax.debug.print("flow_params_obj_eval shape: {}", flow_params_obj_eval.shape)
                
                action = self.sampling_impl.sampling_differ(act_logits, flow_params_obj_eval, key_sample, mask) 
                return action, h_next
            def random_action_fn():
                random_action_normalized = jax.random.uniform(key_rand_act, shape=(max_batches, action_dim), minval=-1.0, maxval=1.0)
                h_reset = self.seq_init(); h_next = jax.tree_map(lambda r, p: jax.lax.select(done, r, p), h_reset, h_prev)
                return random_action_normalized, h_next
            
            # print("etawert", eval_mode)
            
            action, h_next = jax.lax.cond(eval_mode == "eval_random", random_action_fn, ppo_action_fn)
            def step_fn(): return MultiFunctionGymnax.step_env(key_env, env_state, action, env_params, max_batches, action_dim)
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
            next_obs, next_env_state, reward, next_done, info = jax.lax.cond(done, no_step_fn, step_fn)
            reward_accum = jax.lax.select(done, 0.0, reward); new_done = done | next_done
            next_carry = (h_next, next_obs, next_env_state, new_done, key_carry)
            step_output = {"reward": reward_accum, "info": info, "done": new_done}
            return next_carry, step_output

        @partial(jax.vmap, in_axes=(0, 0, 0, 0)) 
        def _run_single_episode(key_episode, h_init, obs_init, env_state_init):
            initial_carry = (h_init, obs_init, env_state_init, jnp.array(False), key_episode)
            _, step_outputs = jax.lax.scan(_eval_step_scan, initial_carry, None, length=max_steps_in_episode)
            total_reward = jnp.sum(step_outputs["reward"]); true_done_idx = jnp.argmax(step_outputs["done"])
            episode_length = jnp.where(jnp.any(step_outputs["done"]), true_done_idx + 1, max_steps_in_episode)
            final_info = jax.tree_map(lambda x: x[episode_length-1], step_outputs["info"])
            success = final_info.get("success", False); regret = final_info.get("scaled_diff", 0.0); best_action = final_info.get("best_rewards", 0.0)
            return {"episode_return": total_reward, "episode_length": episode_length, "best_action": best_action, "regret": regret, "success": success,}

        keys_eval = jax.random.split(key, num_eval_episodes); eval_results_per_episode = _run_single_episode(keys_eval, batch_h_states, batch_obs, batch_env_states)
        eval_metrics_mean = jax.tree_map(jnp.mean, eval_results_per_episode)
        return eval_metrics_mean

    def evaluate(self, key: chex.PRNGKey, params: chex.ArrayTree, env_params_dict: dict, num_eval_episodes: int, action_dim: int, max_batches: int, max_steps_in_episode: int, eval_mode: str = 'ppo') -> dict:
        # ... (code is identical to the version in jax_agent_with_flow) ...
        all_eval_metrics = {}
        env_names = list(env_params_dict.keys())
        
        print("evaluate: paramshh", params.keys())
        
        
        ac_original_params = params['ac']
        flow_obj_original = params.get('flow_dynamic', None)

        # Prepare parameters for the JITted function
        dynamic_params_for_jit = {'ac': ac_original_params}
        static_flow_structure_for_jit: Optional[AbstractBijection] = None

        if flow_obj_original is not None:
            # Partition the flow object
            dynamic_flow_parts, static_structure = eqx.partition(flow_obj_original, eqx.is_array)
            dynamic_params_for_jit['flow_dynamic'] = dynamic_flow_parts
            static_flow_structure_for_jit = self.static_flow_structure
        else:
            # If there's no flow, flow_dynamic can be an empty dict or a specific marker if needed.
            # The JITted function needs to handle static_flow_structure_for_jit being None.
            dynamic_params_for_jit['flow_dynamic'] = {} # Placeholder if no flow
        
        for i, env_name in enumerate(env_names):
            key, subkey = jax.random.split(key) 
            current_env_params = env_params_dict[env_name]
            eval_metrics_single_type = self.evaluate_func_type(subkey, dynamic_params_for_jit, static_flow_structure_for_jit, current_env_params, num_eval_episodes, action_dim, max_batches, max_steps_in_episode, eval_mode)
            for metric_name, value in eval_metrics_single_type.items():
                all_eval_metrics[f"eval_{env_name}/{metric_name}"] = value
        return all_eval_metrics


# --- Helper Functions ---
# (l2_norm and calculate_gae_jax remain the same)
def l2_norm(pytree):
    leaves, _ = jax.tree_util.tree_flatten(pytree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))

# def calculate_gae_jax(rewards: chex.Array, dones: chex.Array, values: chex.Array, gamma: float, gae_lambda: float) -> tuple[chex.Array, chex.Array]:
#     # ... (code is identical to the version in jax_agent_with_flow) ...
#     T = rewards.shape[1]
#     def _gae_step(carry, t_data):
#         gae_next, next_value = carry; reward_t, done_t, value_t = t_data 
#         delta = reward_t + gamma * next_value * (1.0 - done_t) - value_t
#         gae_t = delta + gamma * gae_lambda * (1.0 - done_t) * gae_next
#         return (gae_t, value_t), gae_t
#     rewards_rev = jnp.flip(rewards, axis=1); dones_rev = jnp.flip(dones, axis=1)
#     values_T_rev = jnp.flip(values[:, :-1], axis=1) 
#     scan_inputs = (rewards_rev, dones_rev, values_T_rev)
#     initial_carry = (jnp.zeros(rewards.shape[0]), values[:, -1])
#     _, advantages_rev = jax.lax.scan(_gae_step, initial_carry, scan_inputs, length=T, unroll=16)
#     advantages = jnp.flip(advantages_rev, axis=1)
#     targets = advantages + values[:, :-1] 
#     return advantages, targets
