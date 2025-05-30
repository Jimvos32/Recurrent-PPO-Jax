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
from src.models.actor_critic import ActorCriticPredictor # Import the updated ActorCriticModelPrediction
# Assume FlowMVNJax is defined as in the previous response (ID: flowjax_updated_sampler_v2)
# from .flow_jax import FlowMVNJax 

from src.agents.normalising_flow.flow_construction import create_flow_chain
from flowjax.bijections import Tanh

import numpy as np
# from src.tasks.envs.jax_env_f.jax_function_samplers import compute_y_sampler_dispatch
from src.tasks.envs.jax_env_f.jax_disp_samplers import compute_y_sampler_dispatch
from src.tasks.envs.jax_env_f.jax_env import get_obs, map_to_bounds_jax, get_info
import matplotlib.pyplot as plt
from src.model_fns.pred_fns import PredictorModel



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
    episode_length: chex.Array 
   
    masks: Optional[chex.Array] = None 
    


class NormPPOAgentJaxPred:
    def __init__(self,
                 env_params: EnvParams,
                 env_params_test: EnvParams,
                 repr_model_fn: Callable,
                 seq_model_fn: tuple[Callable, Callable],
                 actor_fn: Callable,
                 critic_fn: Callable,
                 predictor_fn: PredictorModel,
                 optimizer: optax.GradientTransformation,
                 sampling_impl_class, # e.g., FlowMVNJax
                 # --- Flow Configuration (List of Layers) ---
                 # List of tuples: [(LayerClass1, layer_kwargs1), (LayerClass2, layer_kwargs2), ...]
                 flow_layer_configs: List[Tuple[Type[AbstractBijection], Dict[str, Any]]] | None = None,
                 pred_flow_layer_configs: List[Tuple[Type[AbstractBijection], Dict[str, Any]]] | None = None,
                 # --- PPO Hyperparameters ---
                 rollout_len: int = 128,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 num_minibatches: int = 4,
                 update_epochs: int = 4,
                 norm_adv: bool = True,
                 clip_coef: float = 0.1,
                 ent_coef_schedule: optax.Schedule = lambda _: 0.01,
                 value_coef_schedule: optax.Schedule = lambda _: 0.5, # Use a schedule for vf_coef
                 pred_coef_schedule: optax.Schedule = lambda _: 0.5, # Use a schedule for pred_coef
                 pol_coef_schedule: optax.Schedule = lambda _: 0.5, # Use a schedule for pred_coef
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: Optional[float] = None,
                 reset_frequency: int = 5,
                 lstm_hidden_size: int = 64, # Size of the LSTM hidden state
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
        self.value_coef_schedule = value_coef_schedule  # Use a schedule for vf_coef
        self.pred_coef_schedule = pred_coef_schedule  # Use a schedule for pred_coef
        self.pol_coef_schedule = pol_coef_schedule  # Use a schedule for pol_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        # ... (rest of hyperparameter initializations) ...
        self.optimizer = optimizer
        self.reset_frequency = reset_frequency
        self.lstm_hidden_size = lstm_hidden_size
        
        self.action_dim = env_params.action_dim 

        # --- Sampling Implementation (e.g., FlowMVNJax) ---
        self.sampling_impl = sampling_impl_class(
            action_dim=self.action_dim,
            max_batch=env_params.max_batches,
        )
        
        self.prediction_sampler = sampling_impl_class(
            action_dim=1,
            max_batch=1,
        )
        
        self.predictor_model = predictor_fn
        self.predictor_apply = self.predictor_model.apply
        

        self.seq_fn, self.seq_init = seq_model_fn

        # --- Build Actor-Critic Model (Flax) ---
        self.ac_model = ActorCriticPredictor(repr_model_fn, self.seq_fn, actor_fn, critic_fn)
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
            self.static_flow_structure = None
            print("AGENT INIT: No flow configured, static_flow_structure is None.")
            
        self.flow_layer_configs_pred = pred_flow_layer_configs if pred_flow_layer_configs else []
        self.static_flow_structure_pred: Optional[AbstractBijection] = None # Initialize attribute
            
            
        temp_flow_obj_pred = create_flow_chain( 
            key=init_key_for_structure_only, # Key here might not matter if layers are deterministic in structure
            action_dim=1, 
            layer_configs=self.flow_layer_configs_pred
        )
        if temp_flow_obj_pred is not None:
            _, self.static_flow_structure_pred = eqx.partition(temp_flow_obj_pred, eqx.is_array)
            print(f"AGENT INIT: Stored static_flow_structure with shape: {self.static_flow_structure_pred.shape}")
        else:
            self.static_flow_structure_pred = None
            print("AGENT INIT: No flow configured, static_flow_structure is None.")
 
        # Flow chain instance will be created and stored in parameters during init

        # --- Define Actor-Critic Step Helper ---
        # (No changes needed from previous version)
        def _actor_critic_step(ac_params, random_key, obs, done, h_prev):
            obs_b = jax.tree_map(lambda x: jnp.expand_dims(x, 0), obs)
            done_b = jnp.expand_dims(done, 0)
            h_prev_b = jax.tree_map(lambda x: x, h_prev) 
            act_logits_b, value_b, h_next_b, hidden = self.ac_apply({'params': ac_params}, obs_b, done_b, h_prev_b)
            act_logits = act_logits_b.squeeze(0)
            value = value_b.squeeze(0)
            return act_logits, value, h_next_b, hidden
        self._actor_critic_step = _actor_critic_step
        
        
        def pred_sampler_integration(ac_params, obs, done, h_prev, dyn_params, mask, key):
            act_logits, value, memory, _ = self._actor_critic_step(ac_params, jax.random.PRNGKey(0), obs, done, h_prev)
            action = self.sampling_impl.sampling_differ(act_logits, dyn_params, key, mask)
            log_prob = self.sampling_impl.gaussian_log_prob(jnp.expand_dims(action, axis=0), jnp.expand_dims(act_logits, axis=0), dyn_params)
            
            # #Here I need to call it
            # # print("hidden shape", hidden.shape, "action shape", action.shape)

            # pred_output = self.predictor_apply( # Use self.predictor_apply here
            #     {'params': pred_params}, # Pass the predictor's own parameters
            #     hidden,                                # First argument to PredictorModel.__call__
            #     action                                 # Second argument to PredictorModel.__call__
            # )
            return act_logits, action, log_prob, value, memory#, pred_output
        
        self.pred_sampler_integration = pred_sampler_integration

    def init(self, key: chex.PRNGKey) -> AgentTrainState:
        """Initializes agent parameters (AC and Flow Chain) and optimizer state."""
        key, ac_key, flow_init_key, pred_key = jax.random.split(key, 4)

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
        
      
        
        
        dummy_hidden_for_pred = jnp.zeros((1, self.lstm_hidden_size)) #This was (1,257) now (1,256)
        dummy_action_for_pred = jnp.zeros((1, self.env_params.action_dim))     
        
        # print("dummy_hidden_for_pred shape:", dummy_hidden_for_pred.shape, "dummy_action_for_pred shape:", dummy_action_for_pred.shape)
        
        predictor_params = self.predictor_model.init(
            {'params': pred_key}, dummy_hidden_for_pred, dummy_action_for_pred
        )['params']
        
        dynamic_params_for_state = {'ac': ac_params, 'predictor': predictor_params}
        

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
            
            
        if self.static_flow_structure_pred is not None: # Check if flow is configured
            # Create the initial flow object with actual parameters using flow_init_key
            actual_flow_obj = create_flow_chain( 
                key=flow_init_key, 
                action_dim=int(1),
                layer_configs=self.flow_layer_configs_pred
            )
            if actual_flow_obj is not None:
                dynamic_flow_parts, _ = eqx.partition(actual_flow_obj, eqx.is_array)
                dynamic_params_for_state['pred_flow_dynamic'] = dynamic_flow_parts
            else: # Should not happen if self.static_flow_structure is not None
                dynamic_params_for_state['pred_flow_dynamic'] = {} 
        else:
            dynamic_params_for_state['pred_flow_dynamic'] = {} # Placeholder if no flow
            
            

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
        
        # print("Rollout called with keys shape:", dynamic_params["predictor"]['MLP_0'].keys())
        ac_params_rollout = dynamic_params['ac']
        predictor_params_rollout = dynamic_params['predictor']
        
        
        # jax.debug.print("does this even change {}", predictor_params_rollout['MLP_0']['Dense_0']['kernel'][0,0])
        
        dynamic_flow_parts_rollout = dynamic_params.get('flow_dynamic', {})

        def _env_step_scan(carry, key_t):
            h_prev, obs, env_state, last_done = carry 
            key_model, key_sample, key_env_step, key_env_reset = jax.random.split(key_t, 4)
            
            current_flow_obj_rollout = None
            if self.static_flow_structure is not None: # Use agent's static structure
                current_flow_obj_rollout = eqx.combine(dynamic_flow_parts_rollout, self.static_flow_structure)
                
            h_reset = self.seq_init() 
            h_processed = jax.tree_map(lambda reset_h, prev_h: jax.lax.select(last_done, reset_h, prev_h), h_reset, h_prev)
            # act_logits, value, h_next = self._actor_critic_step(ac_params_rollout, key_model, obs, last_done, h_processed) 
            mask = obs.get("mask", None)
            
            #  # Pass the Chain object here
            # action = self.sampling_impl.sampling_differ(act_logits, current_flow_obj_rollout, key_sample, mask)
            # log_prob = self.sampling_impl.gaussian_log_prob(jnp.expand_dims(action, axis=0), jnp.expand_dims(act_logits, axis=0), current_flow_obj_rollout)
            
            act_logits, action, log_prob, value, h_next = self.pred_sampler_integration(
                ac_params_rollout, obs, last_done, h_processed, current_flow_obj_rollout, mask, key_sample)
            
            
            # print("you must not forget !!!!!!!!")
            # pred = action
            
            # log_prob = log_prob.squeeze(0) 
            # ... (rest of _env_step_scan logic) ...
            current_obs, current_env_state, reward, done, info = MultiFunctionGymnax.step_env(key_env_step, env_state, action, env_params, max_batches, action_dim)
            
            # jax.debug.print("action {}, obs {}\nact3 {} obs3 {}\n", action, obs["observations"], current_obs["actions"], current_obs["observations"]) # Debug print to check action and observation shapes

            
            def reset_same_fn(_): return MultiFunctionGymnax.reset_env_keep(key_env_reset, env_params, action_dim, max_batches, current_env_state)
            def reset_new_fn(_): return MultiFunctionGymnax.reset_env(key_env_reset, env_params, action_dim, max_batches)
            def no_reset_fn(_): return current_obs, current_env_state
            reset_mode = jnp.array(1); 
            reset_mode = jnp.where(env_state.episode_counter >= self.reset_frequency - 1, 2, reset_mode)
            # was_successful = info.get("success", jnp.array(False))
            # reset_mode = jnp.where(was_successful, 2, reset_mode)
            final_mode = jax.lax.select(done, reset_mode, 0)
            next_obs_carry, next_env_state_carry = jax.lax.switch(final_mode, [no_reset_fn, reset_same_fn, reset_new_fn], operand=None)
            step_data = {"obs": obs, "action": action, "reward": reward, "done": done, "value": value, "log_prob": log_prob, 
                         "actor_pred": act_logits, "mask": mask, "next_obs": current_obs, "hidden_state": h_processed, 
                         "start_dones": last_done, "success": info["success"], "regret": info["scaled_diff"], 
                         "best_rewards": info["best_rewards"], "last_yes": info["episode_length"], 
                         "tick": info["episode_length"]}
            next_carry = (h_next, next_obs_carry, next_env_state_carry, done)
            return next_carry, step_data

        def _rollout_single_env(key_env, h_init, obs_init, env_state_init):
            initial_carry = (h_init, obs_init, env_state_init, jnp.array(False, dtype=bool))
            keys_t = jax.random.split(key_env, self.rollout_len) 
            (final_h, final_obs, final_env_state, final_done), step_data_sequence = jax.lax.scan(_env_step_scan, initial_carry, keys_t, length=self.rollout_len)
            key_final_model, _ = jax.random.split(key_env)
            h_reset_final = self.seq_init()
            final_h_processed = jax.tree_map(lambda reset_h, prev_h: jax.lax.select(final_done, reset_h, prev_h), h_reset_final, final_h)
            _, final_value, _, _ = self._actor_critic_step(ac_params_rollout, key_final_model, final_obs, final_done, final_h_processed)
            all_obs = jax.tree_map(lambda init, seq_next: jnp.concatenate([jnp.expand_dims(init, 0), seq_next], axis=0), obs_init, step_data_sequence["next_obs"])
            all_values = jnp.concatenate([step_data_sequence["value"], jnp.expand_dims(final_value,0)], axis=0)
            rollout = RolloutData(observations=all_obs, actions=step_data_sequence["action"], rewards=step_data_sequence["reward"], 
                                  dones=step_data_sequence["done"], log_probs=step_data_sequence["log_prob"], values=all_values,
                                  actor_preds=step_data_sequence["actor_pred"], masks=step_data_sequence["mask"], 
                                  hidden_states=step_data_sequence["hidden_state"], start_dones=step_data_sequence["start_dones"],
                                  success=step_data_sequence["success"], regret=step_data_sequence["regret"], 
                                  best_actions=step_data_sequence["best_rewards"], last_step=step_data_sequence["last_yes"], 
                                  episode_length=step_data_sequence["tick"])
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
        
        
        # jax.debug.print("actions {} \nobs {}", actions_T[0, :5], obs_T["observations"][0, :5]) # Debug print to check actions and observations

       
        
        # jax.debug.print("preds {}\nacts {}\noobs {}\nloss {}\n", predictions_T[0,:5].squeeze(), actions_T[0, :5].squeeze(), obs_T["observations"][0, :5].squeeze(), pred_loss[0,:5].squeeze())
       
        
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
        
        #Why is this done?? Maybe not do this?? and slice in the loss function instead?
        # Observations s_0 to s_{T-1} needed for loss calc
        # observations_loss = jax.tree_map(lambda x: x[:, :-1], obs_T) # Shape (B, T, *)
        
        # print("obs", observations_loss["observations"].shape, obs_T["observations"].shape)
        
        # Initial hidden state h_0 for each sequence
        hidden_states_init = jax.tree_map(lambda x: x[:, 0], hidden_states_T) # Shape (B, *h)
        
        actions_T = rollout_data.actions; log_probs_T = rollout_data.log_probs; masks_T = rollout_data.masks
        if self.norm_adv: advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def _ppo_loss_fn_single(params_for_loss_calc: chex.ArrayTree, random_key, obs_seq, actions_seq, logp_old_seq, 
                                adv_seq, targets_seq, masks_seq, start_dones_seq, h_init):
            key_model, key_entropy = jax.random.split(random_key)
            ac_params_loss = params_for_loss_calc['ac']
            pred_params_loss = params_for_loss_calc['predictor']
        
            reconstructed_flow_obj_loss = None
            if self.static_flow_structure is not None: # From agent instance
                # params_for_loss_calc['flow_dynamic'] has the current dynamic flow params
                # jax.debug.print("flow_dynamic params shape: {} {} {}", params_for_loss_calc['flow_dynamic'][0].params,  params_for_loss_calc['flow_dynamic'][-1].params, ac_params_loss['actor']['MLP_1']['Dense_0'])
                reconstructed_flow_obj_loss = eqx.combine(
                    params_for_loss_calc.get('flow_dynamic', {}), # Default to empty if not present
                    self.static_flow_structure
                )
            
            
            obs_seq_b = jax.tree_map(lambda x: x[:-1], obs_seq) # Shape (B, T, *)
            
            
            # jax.debug.print("obs_seq_b shape: {}\nactions_seq shape: {}",
            #                 jnp.sqrt(obs_seq_b["observations"][-5:]), jnp.squeeze(obs_seq["observations"][-5:]))
          
            # Recompute AC outputs
            ##Potentially slice here??
            act_logits_seq, values_seq, _, hidden = self.ac_apply({'params': ac_params_loss}, obs_seq_b, start_dones_seq, h_init, seq_grad=False)
            # values_seq = values_seq.squeeze(-1) 
            # Recompute log probs and entropy, passing the Chain object
            logp_new_seq = self.sampling_impl.gaussian_log_prob(actions_seq, act_logits_seq, reconstructed_flow_obj_loss) 
            
            # print("logp_new_seq shape", act_logits_seq.shape, "logp_old_seq shape", logp_old_seq.shape)
            
            entropy_seq = self.sampling_impl.entropy(act_logits_seq, masks_seq, reconstructed_flow_obj_loss, key=key_entropy) 
            mean_entropy = entropy_seq.mean() 
            

            print("action", actions_seq.shape, "hidden", hidden.shape)
            
            # batch_prediction = jax.vmap(
            #     lambda single_action: self.predictor_apply(
            #         {'params': pred_params_loss},
            #         hidden,
            #         single_action # Reshape (D,) to (1, D)
            #     ),
            #     in_axes=1 # We are mapping over the rows of x_policy_pdf_norm_jax
            # )

            # # 3. Get all predictions in one go
            # pred_output = batch_prediction(actions_seq)
            
            # print("act_s shape", act_s.shape, "hidden shape", hidden.shape, pred_output.shape)
            
            
            # act_s = jnp.squeeze(actions_seq, axis=-1) 
       
            # print("act_s shape", act_s.shape, "hidden shape", hidden.shape, pred_output.shape)
            
            
                
            # pred_output = self.predictor_apply( # Use self.predictor_apply here
            #     {'params': pred_params_loss}, # Pass the predictor's own parameters
            #     hidden,                                # First argument to PredictorModel.__call__
            #     act_s                                 # Second argument to PredictorModel.__call__
            # )
            
            # print("act_s shape", act_s.shape, "hidden shape", hidden.shape, pred_output.shape)
            print("2act_s shape", actions_seq.shape)
            batch_prediction = jax.vmap(
                lambda single_action: self.predictor_apply(
                    {'params': pred_params_loss},
                    hidden,
                    single_action # Reshape (D,) to (1, D)
                ),
                in_axes=1 # We are mapping over the rows of x_policy_pdf_norm_jax
            )
           
            # 3. Get all predictions in one go
            pred_output = batch_prediction(actions_seq)
            
            print("2act_s shape", actions_seq.shape, "h2idden shape", hidden.shape, pred_output.shape)
            
            
            pred_output = jnp.transpose(pred_output, (1, 0, 2))  # Transpose to match the expected shape
            
            
            
            
            
            # Policy Loss
            logratio = logp_new_seq - logp_old_seq
            ratio = jnp.exp(jnp.clip(logratio, -20, 20))
            pg_loss1 = -adv_seq * ratio; pg_loss2 = -adv_seq * jnp.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
            # Value Loss
            v_coeff = self.value_coef_schedule(update_step)  # Use the schedule for vf_coef
            v_loss =  v_coeff * jnp.square(values_seq - targets_seq).mean()
            # Total Loss
            ent_coef = self.ent_coef_schedule(update_step)
        
            
            # jax.debug.print("ent_coef {}", ent_coef)
            ent_loss = ent_coef * mean_entropy
            
            
            # jax.debug.print("pg_loss {}, v_loss {}, ent_loss {}, ent_coef {}", pg_loss, v_loss, ent_loss, ent_coef)
            
            pred_coeff = self.pred_coef_schedule(update_step)
            # print("predicitons_seq shape", pred_loss_seq.shape, "obs_seq shape", obs_seq["observations"].shape, actions_seq.shape)
            
            # print("pred_loss_seq shape", pred_output.shape, "actions_seq shape", obs_seq["observations"].shape)
            
            ####This is the part where I have to cut data from both sides of the sequence, which seems to be wasting data
            
            pred_flow_object = None
            if self.static_flow_structure_pred is not None: # From agent instance
                # params_for_loss_calc['flow_dynamic'] has the current dynamic flow params
                # jax.debug.print("flow_dynamic params shape: {} {} {}", params_for_loss_calc['flow_dynamic'][0].params,  params_for_loss_calc['flow_dynamic'][-1].params, ac_params_loss['actor']['MLP_1']['Dense_0'])
                pred_flow_object = eqx.combine(
                    params_for_loss_calc.get('pred_flow_dynamic', {}), # Default to empty if not present
                    self.static_flow_structure_pred
                )
            
            # prediction = self.prediction_sampler.sampling_differ(
            #     pred_output, pred_flow_object, key_model, masks_seq)
            
            # pred_logprob = self.sampling_impl.gaussian_log_prob(actions_seq, act_logits_seq, reconstructed_flow_obj_loss) 
            
            
            prediction_loss = jax.vmap(
                lambda true_obs, dist_params: self.prediction_sampler.gaussian_log_prob(
                    jnp.expand_dims(true_obs, 1),
                    jnp.expand_dims(dist_params, 1),
                    pred_flow_object # Reshape (D,) to (1, D)
                ),
                in_axes=(1,1) # We are mapping over the rows of x_policy_pdf_norm_jax
            )
           
            # 3. Get all predictions in one go
            pred_losses = prediction_loss(obs_seq["observations"][1:], pred_output)
            # pred = obs_seq["observations"][1:]
            # print("pred shape", pred.shape, "pred_output shape", pred_output.shape, "obs_seq shape", obs_seq["observations"].shape)
            # gaussian_log_prob_pred = self.prediction_sampler.gaussian_log_prob(
            #     obs_seq["observations"][1:], pred_output, pred_flow_object)
            
            # print("pred shape", pred.shape, "pred_output shape", pred_output.shape, "obs_seq shape", obs_seq["observations"].shape, "gaussian_log_prob_pred shape", gaussian_log_prob_pred.shape)
            
            
            
            # pred_diff = pred_output - jnp.squeeze(obs_seq["observations"][1:], -1)  # Predictions minus observations
            print("pred_losses shape", pred_losses.shape, "pred_output shape", pred_output.shape, "obs_seq shape", obs_seq["observations"].shape)
         
            pred_loss = pred_coeff * jnp.mean(jnp.square(pred_losses))  # Mean squared error for predictions
            
            # print("pred_act", actions_seq[:5].shape, "pred_obs", obs_seq["observations"][:5].shape, predicitons_seq[:5].shape)  # Debug print to check actions and observations
            
            # jax.debug.print("pred_act {}\npred_obs {}\nnons_obs {}\npred_pre {}\naa {}\nbb {}\n", jnp.squeeze(actions_seq[-5:,:,:]), jnp.squeeze(obs_seq["observations"][-5:])
            #                 ,jnp.squeeze(seq_oobs["observations"][-5:]), jnp.squeeze(pred_output[-5:]), pred_loss_seq[-5:], pred_loss_ll[-5:])  # Debug print to check actions and observations
            pol_coeff = self.pol_coef_schedule(update_step)  # Use the schedule for pol_coef
            pg_loss = pol_coeff * pg_loss  # Scale policy loss by pol_coef
            
            # jax.debug.print("v_coef {}, ent_coef {}, pred_coeff {}, pol_coeff {}", v_coeff, ent_coef, pred_coeff, pol_coeff)
            
            total_loss = pg_loss + v_loss - ent_loss + pred_loss
            # total_loss = pred_loss
            # Metrics
            approx_kl = jnp.mean((ratio - 1) - logratio)
            metrics = {"loss/loss": total_loss, "loss/loss_policy": pg_loss, "loss/loss_value": v_loss, "loss/loss_entropy": ent_loss, "loss/kl_approx": approx_kl, "loss/ratio": jnp.mean(ratio), "loss/log_old": jnp.mean(logp_old_seq), "loss/log_new": jnp.mean(logp_new_seq), "loss/values_pred": jnp.mean(values_seq), 
                       "loss/target": jnp.mean(targets_seq), "loss/entropy_coefficient": ent_coef, "loss/pred_loss": pred_loss}
            return total_loss, metrics

        def _update_epoch(carry, _):
            agent_state_epoch, key_epoch = carry
            key_epoch, key_perm = jax.random.split(key_epoch); perms = jax.random.permutation(key_perm, num_envs)
            def shuffle_env_dim(x): return None if x is None else jax.tree_map(lambda leaf: leaf[perms], x)
            observations_shuffled = shuffle_env_dim(obs_T); actions_shuffled = shuffle_env_dim(actions_T)
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
                (mean_loss, batch_metrics), grads = grad_target_fn(agent_state_mb.params, key_loss, mb_obs_seq, mb_actions_seq, 
                                                                   mb_logp_old_seq, mb_adv_seq, mb_targets_seq, mb_masks_seq, 
                                                                   mb_start_dones_seq, mb_h_init)
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
                act_logits, _, h_next, hidden = self._actor_critic_step(ac_params_eval, key_model, obs, done, h_prev)
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
    
    
    
    
    def run_episode_for_visualization(
        self,
        key: chex.PRNGKey,
        f_name: str,
        agent_state_params: chex.ArrayTree, # Agent's current learnable parameters (agent_state.params)
        env_params_to_visualize: EnvParams, # EnvParams for the specific function type
        num_true_func_points: int = 200,
        num_policy_points: int = 200, 
        
    ) -> Dict:
        """
        Runs a single episode, collects detailed per-step data for visualization,
        and prepares it for plotting. This function is NOT JITted.
        Assumes action_dim = 1 for simple 1D plotting.
        """
        if self.action_dim != 1:
            print(f"Warning: Visualization is best for 1D action space. Current: {self.action_dim}D.")
            # For multi-D, plotting will need to select a dimension or use different techniques.

        print(f"Running single episode for visualization with function: {env_params_to_visualize.function_type_indices}") # Example log

        # --- Initialize Environment for this single run ---
        # Ensure we are using a single environment instance for this
        env = MultiFunctionGymnax() # Create a new env instance for this run
        key_reset, key_episode_steps = jax.random.split(key)

        # Reset the environment using the specific EnvParams provided
        # Make sure action_dim and max_batches are correctly passed for this env_params_to_visualize
        # These might come from env_params_to_visualize itself if it's fully configured,
        # or from self.env_params as defaults if appropriate.
        # For this example, let's assume env_params_to_visualize has necessary fields
        # or we use some defaults from the agent's primary env_params.
        action_dim_vis = env_params_to_visualize.action_dim
        max_batches_vis = env_params_to_visualize.max_batches

        obs, current_env_state = env.reset_env(
            key_reset, env_params_to_visualize, action_dim_vis, max_batches_vis
        )
        current_h_state = self.seq_init() # Initial hidden state
        # If seq_init returns a batched h_state, take the first one
        # current_h_state = jax.tree_map(lambda x: x[0] if x.ndim > 1 and x.shape[0] > 1 else x, current_h_state)


        # --- Prepare for data collection ---
        episode_visualization_data = []
        done = False
        current_tick = 0
        
        all_samples_x_history_list = []
        all_samples_y_history_list = []
        

        # print("env_params_to_visualize", current_env_state.params_for_compute['specific'].keys())
        # print("env_params_to_visualize", current_env_state.params_for_compute['specific'][f_name]['bounds'])
        bounds = env_params_to_visualize.sampler_configs['specific'][f_name]['bounds']
        
        # --- Get True Function Data (once per episode) ---
        x_true_np = np.linspace(
            bounds[0],
            bounds[1],
            num_true_func_points
        ).reshape(-1, action_dim_vis)
        y_true_np = np.asarray(
            compute_y_sampler_dispatch(
                jnp.array(x_true_np),
                current_env_state.params_for_compute, # From initial reset
                env_params_to_visualize
            )
        ).squeeze()
   
        x_true_np = x_true_np.squeeze()


        # --- Reconstruct Flow Object (once, if params don't change during this vis run) ---
        ac_params = agent_state_params['ac']
        pred_params = agent_state_params['predictor']
        dynamic_flow_parts = agent_state_params.get('flow_dynamic', {})
        dynamic_pred_flow_parts = agent_state_params.get('pred_flow_dynamic', {})
        current_policy_flow_object = None
        if self.static_flow_structure is not None:
            if dynamic_flow_parts:
                current_policy_flow_object = eqx.combine(dynamic_flow_parts, self.static_flow_structure)
            else:
                current_policy_flow_object = self.static_flow_structure
                
        pred_flow_obj = None
        if self.static_flow_structure_pred is not None:
            if dynamic_pred_flow_parts:
                pred_flow_obj = eqx.combine(dynamic_pred_flow_parts, self.static_flow_structure_pred)
            else:
                pred_flow_obj = self.static_flow_structure



        # --- Run the episode step-by-step ---
        while not done:

            key_step, key_episode_steps = jax.random.split(key_episode_steps)
            key_actor, key_sample, key_env_step = jax.random.split(key_step, 3)

            # Get action from policy
            # Note: current_env_state.done is from the *previous* step
            act_logits, value_pred, next_h_state, hidden = self._actor_critic_step(
                ac_params, key_actor, obs, current_env_state.done, current_h_state
            )
            
            #  def pred_sampler_integration(ac_params, obs, done, h_prev, dyn_params, mask, key):
            # act_logits, value, memory, hidden = self._actor_critic_step(ac_params, jax.random.PRNGKey(0), obs, done, h_prev)
            # action = self.sampling_impl.sampling_differ(act_logits, dyn_params, key, mask)
            # log_prob = self.sampling_impl.gaussian_log_prob(jnp.expand_dims(action, axis=0), jnp.expand_dims(act_logits, axis=0), dyn_params)
            # pred = self.ac_model.predction(hidden, action)  # Call the prediction method
            # return action, log_prob, value, memory, pred
        
            # self.pred_sampler_integration = pred_sampler_integration
            
            
            
            
            
            

            # Use the agent's sampling implementation
            # The mask for sampling_differ should correspond to current_env_state.batch_size
            # For a single episode run, often batch_size is 1 or a small fixed number.
            # get_obs(current_env_state, env_params_to_visualize)['mask'] gives current batch size
            sampling_mask = get_obs(current_env_state, env_params_to_visualize)['mask']

            action = self.sampling_impl.sampling_differ(
                act_logits, current_policy_flow_object, key_sample, sampling_mask
            )
            
            current_hist_x = np.concatenate(all_samples_x_history_list) if all_samples_x_history_list else np.array([])
            current_hist_y = np.concatenate(all_samples_y_history_list) if all_samples_y_history_list else np.array([])

            # Store data for this step *before* stepping the environment
            step_data_for_plot = {
                "step": current_tick,
                "x_true": x_true_np,
                "y_true": y_true_np,
                "sampler_info": current_env_state.params_for_compute.get('common', {}).get('sampler_type_index', -1),
                "act_logits_numpy": np.asarray(act_logits),
                "all_previous_samples_x_numpy": current_hist_x, # Store history so far
                "all_previous_samples_y_numpy": current_hist_y,
            }
            # Step the environment
            next_obs, next_env_state, reward, done, info = env.step_env(
                key_env_step, current_env_state, action, env_params_to_visualize,
                max_batches_vis, action_dim_vis
            )

            # Store samples *after* stepping (these are the results of `action`)
            # Use next_env_state as it contains last_action_mapped and last_raw_obs from this step
            valid_samples_mask_after_step = jnp.arange(max_batches_vis) < next_env_state.batch_size
        
            current_step_samples_x_np = np.asarray(
                next_env_state.last_action_mapped[valid_samples_mask_after_step]
            ).squeeze()
            current_step_samples_y_np = np.asarray(
                next_env_state.last_raw_obs[valid_samples_mask_after_step]
            ).squeeze()
            
            step_data_for_plot["current_samples_x_numpy"] = current_step_samples_x_np
            step_data_for_plot["current_samples_y_numpy"] = current_step_samples_y_np
            
            # Add current step's samples to history *after* storing for this step's plot
            if current_step_samples_x_np.size > 0:
                # Ensure they are 1D arrays before appending for consistent concatenation
                all_samples_x_history_list.append(current_step_samples_x_np.flatten())
                all_samples_y_history_list.append(current_step_samples_y_np.flatten())


            # --- Generate Policy PDF Data for this step ---
            # x_values for PDF are in normalized action space [-1, 1]
            x_policy_pdf_norm_np = np.linspace(-1.0 + 1e-5, 1.0 - 1e-5, num_policy_points).reshape(-1, action_dim_vis)
            policy_pdf_values_np = np.asarray(
                self.sampling_impl.get_pdf(
                    act_logits, # From this step
                    current_policy_flow_object,
                    jnp.array(x_policy_pdf_norm_np)
                )
            ).squeeze()
            
            # x_policy_pdf_norm_np = np.linspace(-1.0 + 1e-5, 1.0 - 1e-5, num_policy_points).reshape(-1, action_dim_vis)
            
            # 1. Convert to JAX array
            jnp_base = jnp.array(x_policy_pdf_norm_np)
            # Current shape: (num_policy_points, action_dim_vis)

            # 2. Define and apply the vectorized prediction function
            # This lambda takes a single action row (shape (action_dim_vis,))
            # and reshapes it to (1, action_dim_vis) for the predictor model.
            # `pred_params` and `hidden` are closed over from the surrounding scope.
            # vectorized_predictor_call = jax.vmap(
            #     lambda single_action_row: self.predictor_apply(
            #         {'params': pred_params},
            #         hidden,
            #         single_action_row.reshape(1, -1) # Reshape (D,) to (1, D)
            #     ),
            #     in_axes=0 # We are mapping over the rows of x_policy_pdf_norm_jax
            # )

            # # 3. Get all predictions in one go
            # function_estimate_jax = vectorized_predictor_call(jnp_base)
            
           
            
            def predictor_mapping(action, key_env):
                
                print("action shape", action.shape, "hidden shape", key_env.shape, hidden.shape)
                pred_logits = self.predictor_apply(
                    {'params': pred_params},  # Pass the predictor's own parameters
                    hidden,
                    jnp.expand_dims(action, 0))
                pred_samples = self.prediction_sampler.sampling_differ(
                    pred_logits,
                    pred_flow_obj,
                    key_env,
                    sampling_mask)
                return pred_samples
                
            

            predictor_mapping = jax.vmap(predictor_mapping, in_axes=(0, 0))
            
            

            keys_for_vmap = jax.random.split(key_env_step, num_policy_points)

            
            print("predictor_mapping shape", jnp_base.shape, "jnp_base shape", jnp_base.shape, "key_env_step shape", key_env_step.shape)
            
            function_estimate_jax = predictor_mapping(jnp_base, keys_for_vmap)
            
            
            print("function_estimate_jax shape", function_estimate_jax.shape, "x_policy_pdf_norm_np shape", x_policy_pdf_norm_np.shape, "policy_pdf_values_np shape", policy_pdf_values_np.shape)
            
            function_estimate_jax = np.asarray(function_estimate_jax)
            
            
            # min_y = env_params_to_visualize.sampler_configs['specific'][f_name]['min_y']
            # max_y = env_params_to_visualize.sampler_configs['specific'][f_name]['max_y']
            
            min_y = current_env_state.params_for_compute['common']['min_y']
            max_y = current_env_state.params_for_compute['common']['max_y']
            
            
            
            
            function_estimate_jax = function_estimate_jax * (max_y - min_y) + min_y  # Map back to original bounds
            
            function_estimate_jax = function_estimate_jax.squeeze()  # Remove any extra dimensions
            
            # print("function_estimate_jax shape", function_estimate_jax.shape)
            
            # function_estimate = []
            # for i in x_policy_pdf_norm_np:
            #     pred_output = self.predictor_apply( # Use self.predictor_apply here
            #         {'params': pred_params}, # Pass the predictor's own parameters
            #         hidden,                                # First argument to PredictorModel.__call__
            #         i                                 # Second argument to PredictorModel.__call__
            #     )
            #     function_estimate.append(pred_output)
                

            x_policy_pdf_mapped_np = np.asarray(
                map_to_bounds_jax(
                    jnp.array(x_policy_pdf_norm_np),
                    bounds,
                )
            ).squeeze()

            step_data_for_plot["x_policy_mapped_numpy"] = x_policy_pdf_mapped_np
            step_data_for_plot["policy_pdf_numpy"] = policy_pdf_values_np
            step_data_for_plot["function_estimate_numpy"] = function_estimate_jax

            episode_visualization_data.append(step_data_for_plot)

            # Update states for next iteration
            obs = next_obs
            current_env_state = next_env_state
            current_h_state = next_h_state
            current_tick += 1

        return episode_visualization_data # List of dictionaries, one per step


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




def plot_policy_diagnostics(
    x_true: np.ndarray,
    y_true: np.ndarray,
    samples_x: np.ndarray,
    samples_y: np.ndarray,
    x_policy_mapped: np.ndarray,
    policy_pdf: np.ndarray,
    sampler_info: str = "Unknown Function",
    title_suffix: str = ""
):
    """
    Plots the true function, agent samples, and policy distribution.

    Args:
        x_true: X-values for the true function.
        y_true: Y-values for the true function.
        samples_x: X-values of samples taken by the agent.
        samples_y: Y-values of samples taken by the agent.
        x_policy_mapped: X-values for the policy PDF, mapped to the function's domain.
        policy_pdf: PDF values of the policy.
        sampler_info: Name of the true function (e.g., Ackley).
        title_suffix: Optional suffix for the plot title.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot the true function
    color_true_func = 'tab:blue'
    ax1.set_xlabel('x')
    ax1.set_ylabel('True Function Value', color=color_true_func)
    ax1.plot(x_true, y_true, label=f'True Function ({sampler_info})', color=color_true_func, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color_true_func)

    # Plot the samples taken by the agent
    ax1.scatter(samples_x, samples_y, label='Agent Samples', color='red', marker='o', s=50, alpha=0.7, zorder=5)

    # Create a second y-axis for the policy PDF
    ax2 = ax1.twinx()
    color_policy_pdf = 'tab:green'
    ax2.set_ylabel('Policy PDF', color=color_policy_pdf)
    ax2.plot(x_policy_mapped, policy_pdf, label='Policy PDF', color=color_policy_pdf, linewidth=2)
    ax2.fill_between(x_policy_mapped, policy_pdf, color=color_policy_pdf, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color_policy_pdf)

    # Titles and legends
    plt.title(f'RL Agent Diagnostics: {sampler_info}{title_suffix}')
    fig.tight_layout() # Otherwise the right y-label is slightly clipped
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.grid(True, linestyle=':', alpha=0.7)
    # plt.show()
    return plt
    
    
    
