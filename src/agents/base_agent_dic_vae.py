import jax
import jax.numpy as jnp
import optax
import rlax
import tqdm
import numpy as np
import logging

from src.models.actor_critic import *
from typing import Callable
from src.utils import tree_index
from src.models.actor_critic import ActorCriticModel
from argparse import Namespace

logger = logging.getLogger(__name__)

def jax_to_numpy(*args):
    return jax.tree_map(lambda x: np.array(x),args)

def numpy_to_jax(*args,dtype=jnp.float32):
    return jax.tree_map(lambda x: jnp.array(x,dtype=dtype),args)


class BaseAgentDicVAE:
    def __init__(self,train_envs,eval_env,rollout_len,repr_model_fn:Callable,seq_model_fn:Callable,
                        actor_fn:Callable,critic_fn:Callable,use_gumbel_sampling=False,sequence_length=None, continious_sampling=True, single_dim=False, task_name=None) -> None:
        self.env=train_envs
        self.eval_env=eval_env
        self.rollout_len=rollout_len
        if sequence_length is None:
            self.sequence_length=self.rollout_len
        else:
            assert rollout_len%sequence_length==0 
            self.sequence_length=sequence_length
        self.seq_fn,self.seq_init=seq_model_fn
        self.use_gumbel_sampling=use_gumbel_sampling
        self.continious_sampling = continious_sampling
        self.single_dim = single_dim
        self.task = task_name
        self.ac_model=nn.vmap(ActorCriticVAEModel,
                              variable_axes={'params': None},
                                split_rngs={'params': False, 'vae_sample': True})(repr_model_fn,self.seq_fn,actor_fn,critic_fn)
        
        @jax.jit
        def actor_critic_fn(random_key,params,inputs,terminations,last_memory):
            """

            Args:
                random_key (_type_): _description_
                params (_type_): _description_
                inputs (_type_): shape (BXTXrepr_dim)
                last_memory (_type_): _description_

            Returns:
                _type_: _description_
            """
            random_key, vae_sample_key = jax.random.split(random_key)

            act_logits,values,memory,latent=self.ac_model.apply(params,inputs,terminations,last_memory,rngs={'random': random_key, 'vae_sample': vae_sample_key})
            return act_logits,values,memory, latent
        
        
        self.actor_critic_fn=actor_critic_fn
        

        
     
    
    

    def reset(self,params_key,random_key):
        #Reset the Agent and initilize the parameters
        self.tick=0
        self.o_tick,_=self.env.reset()
        self.r_tick=jnp.zeros(self.env.num_envs)
        self.term_tick=jnp.full((self.env.num_envs),False)
        self.h_tickminus1=jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x,axis=0),self.env.num_envs,axis=0),self.seq_init())
        
        expanded_o = self.expand_o_tick(self.o_tick)
    
        self._params=self.ac_model.init({'params':params_key,'random':random_key},expanded_o,jnp.expand_dims(self.term_tick,1),#,jnp.expand_dims(self.o_tick,1),jnp.expand_dims(self.term_tick,1),
                                       self.h_tickminus1)
        def params_sum(params):
            return sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape),params)))
        logger.info("Total Number of params: %d"%params_sum(self.params))
        logger.info("Number of params in Seq Model: %d"%params_sum(self.params['params']['seq_model']))
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, value):
        self._params = value
        
    def expand_o_tick(self, o_tick, eval=False):
        if eval:
            return {key: jnp.expand_dims(value, axis=0) for key, value in o_tick.items()}
        return {key: jnp.expand_dims(value, axis=1) for key, value in o_tick.items()}
        
    def stack_dict_obs(self, obs, dictio):
        if len(dictio.keys()) == 0:
            # If dictionary is empty, initialize it with the current observation
            # print("obs", obs["actions"].shape, np.expand_dims(np.array(obs["actions"]), axis=1).shape)
            return {key: np.expand_dims(np.array(value), axis=1) for key, value in obs.items()}
        else:
            # Stack the new observation with existing ones
            for key in obs.keys():
                if key in dictio:
                    
                    # print("key", key, "obs", np.expand_dims(np.array(obs[key]), axis=1).shape, "dictio", dictio[key].shape)
                    # # Append the new observation along the first axis (batch dimension)
                    # print("dict", (dictio[key]).shape, (np.expand_dims(np.array(obs[key]), axis=1).shape))
                    
                    
                    dictio[key] = np.concatenate([dictio[key], np.expand_dims(np.array(obs[key]), axis=1)], axis=1)
                else:
                    # If this key wasn't in the dictionary yet, initialize it
                    # print("key", key, "obs", obs[key].shape)
                    added_step_dim = jnp.expand_dims(obs[key], axis=1)
                    # print("key", key, "obs", obs[key].shape, added_step_dim.shape)
                    dictio[key] = added_step_dim
            
            return dictio
        
    def sampling_differ(self, task, act_logits,random_key, masks=None):
        if task == "batch":
            # act_logits: shape (parallel_env, 1, 2 * action_dim)
            batch_size = self.eval_env.unwrapped.batch_size  # number of samples per env
            action_dim = act_logits.shape[-1] // 2

            act_logits = jnp.repeat(act_logits, batch_size, axis=2)
            means = act_logits[..., :action_dim]
            log_stds = act_logits[..., action_dim:]
            log_stds = jnp.clip(log_stds, -20.0, 2.0)
            stds = jnp.exp(log_stds)

        

            # 4. Sample noise and compute actions:
            noise = jax.random.normal(random_key, shape=means.shape)  # shape: (parallel_env, batch_size, action_dim)
            acts_tick = means + noise * stds  # shape: (parallel_env, batch_size, action_dim)
            # acts_tick = jnp.squeeze(acts_tick)  # shape: (parallel_env, action_dim)
            # print("ac", acts_tick.shape)
            
            
        elif task == "sampling":
            action_dim = act_logits.shape[-1] // 2
            means = act_logits[..., :action_dim].squeeze(-1)
            log_stds = act_logits[..., action_dim:].squeeze(-1)
            # print("policy_out", act_logits.shape, "mean", means.shape, "std", log_stds.shape)
            
            # Clip log_stds for numerical stability
            log_stds = jnp.clip(log_stds, -20.0, 2.0)
            stds = jnp.exp(log_stds)
            
            # Sample from standard normal and scale
            noise = jax.random.normal(random_key, means.shape)
            acts_tick = means + noise * stds
            # jax.debug.print("dit kan echt niet meer {} {} {} ", means.shape, log_stds.shape, acts_tick.shape)
            acts_tick = jnp.squeeze(acts_tick)
            
        elif task == "multibatch":
            batch_size = self.eval_env.unwrapped.batch_size  # number of samples per env
            action_dim = act_logits.shape[-1] // 2 # shape [parallel_env, 1, 2 * action_dim]

            means = act_logits[..., :action_dim]
            log_stds = act_logits[..., action_dim:]
            log_stds = jnp.clip(log_stds, -20.0, 2.0)
            stds = jnp.exp(log_stds)
            
            # print("means", means.shape, "log_stds", log_stds.shape)

            #samples the actions
            noise = jax.random.normal(random_key, shape=means.shape)  # shape: (parallel_env, batch_size, action_dim)
            acts_tick = means + noise * stds  # shape: (parallel_env, batch_size, action_dim)
        elif task == "expanded_samp":  
            # print("pol_output", act_logits.shape)
            means, log_stds, weight_logits = jnp.split(act_logits, 3, axis=-1) # shape: (parallel_env, 1, 3 *k)
            log_stds = jnp.clip(log_stds, -20, 2)
            weights = jax.nn.softmax(weight_logits, axis=-1)  # shape: (parallel_env, 1, k)

            batch_size = self.eval_env.unwrapped.batch_size  # number of samples per environment
            # print("means", means.shape, "log_stds", log_stds.shape, "weight_logits", weight_logits.shape, "weights", weights.shape, act_logits.shape)

            # Get dimensions
            N = means.shape[0]      # number of parallel envs
            k = means.shape[-1]     # number of mixture components
            
            
            # Broadcast parameters from shape (N, 1, k) to (N, batch_size, k)
            means = jnp.broadcast_to(means, (N, batch_size, k))
            log_stds = jnp.broadcast_to(log_stds, (N, batch_size, k))
            weight_logits = jnp.broadcast_to(weight_logits, (N, batch_size, k))
            weights = jnp.broadcast_to(weights, (N, batch_size, k))

            # Split the random key for independent sampling.
            key_cat, key_noise = jax.random.split(random_key)

            # Sample mixture component indices from the logits.
            # jax.random.categorical expects logits of shape (..., num_classes) and returns
            # a sample of shape matching the input without the last dimension.
            # Here, weight_logits has shape (N, batch_size, k) so we get (N, batch_size)
            component_indices = jax.random.categorical(key_cat, logits=weight_logits, axis=-1)

            # Expand indices so they can index the last dimension (the k components)
            component_indices = component_indices[..., None]  # shape: (N, batch_size, 1)

            # Gather the chosen means and log_stds based on the sampled component indices.
            chosen_means = jnp.take_along_axis(means, component_indices, axis=-1)      # shape: (N, batch_size, 1)
            chosen_log_stds = jnp.take_along_axis(log_stds, component_indices, axis=-1)  # shape: (N, batch_size, 1)
            chosen_stds = jnp.exp(chosen_log_stds)

            # Sample noise from a standard normal distribution matching the chosen parameters’ shape.
            noise = jax.random.normal(key_noise, shape=chosen_means.shape)

            # Compute the final sampled actions.
            # acts_tick shape: (parallel_env, batch_size, 1)
            acts_tick = chosen_means + chosen_stds * noise
            # print("acts_tick", acts_tick.shape, )
            
        elif task == "multidim":  
            action_dim = self.eval_env.unwrapped.action_dim  # Ensure your environment defines action_dim

            # Split the policy output into means and log_stds.
            means, log_stds = jnp.split(act_logits, 2, axis=-1)
            
            log_stds = jnp.clip(log_stds, -20, 2)
            stds = jnp.exp(log_stds)

            batch_size = self.eval_env.unwrapped.batch_size  # number of samples per environment

            # Get the number of parallel environments.
            N = means.shape[0]

            # Broadcast parameters to match the batch size.
            # New shape becomes (N, batch_size, action_dim)
            means = jnp.broadcast_to(means, (N, batch_size, action_dim))
            stds = jnp.broadcast_to(stds, (N, batch_size, action_dim))
            # Sample noise from a standard normal distribution matching the shape.
            noise = jax.random.normal(random_key, shape=means.shape)

            # Compute the final sampled actions.
            acts_tick = means + stds * noise

            
            return acts_tick
        
        elif task == "masked":  
            
            def apply_padding_mask(expanded_array, padding_counts, pad_value=-2.0):
                x, n, z = expanded_array.shape
                
                # Create indices for each position in the batch dimension
                batch_indices = jnp.arange(n)
                
                # Reshape padding_counts and broadcast for comparison
                counts_expanded = padding_counts.reshape(x, 1)
                
                # Create a mask where True means we should keep the original value
                # and False means we should pad
                mask = batch_indices < counts_expanded
                
                # Expand the mask to match the full shape
                full_mask = jnp.broadcast_to(mask.reshape(x, n, 1), (x, n, z))
                
                # Apply the mask using where to conditionally select values
                result = jnp.where(full_mask, expanded_array, pad_value)
                
                return result
            
            
            action_dim = self.eval_env.unwrapped.action_dim  # Ensure your environment defines action_dim

            # Split the policy output into means and log_stds.
            means, log_stds = jnp.split(act_logits, 2, axis=-1)
            
            log_stds = jnp.clip(log_stds, -20, 2)
            stds = jnp.exp(log_stds)

            batch_size = self.eval_env.unwrapped.max_batches  # number of samples per environment
            

            # Get the number of parallel environments.
            N = means.shape[0]
            
            # print("means", means.shape, "log_stds", log_stds.shape, "stds", stds.shape) #from (8,1,3)
            # Broadcast parameters to match the batch size.
            # New shape becomes (N, batch_size, action_dim)
            means = jnp.broadcast_to(means, (N, batch_size, action_dim)) #broadcasts to (8, 3, 3), duplicates the 1st dimension
            stds = jnp.broadcast_to(stds, (N, batch_size, action_dim))
            
            # print("asdf", means.shape) #output = #(8,3,3)
            #Should expand into (N, max_size, action_dim) with the extra (max_size - batch_size) padded with zeros
            
            # Sample noise from a standard normal distribution matching the shape.
            noise = jax.random.normal(random_key, shape=means.shape)

            # Compute the final sampled actions.
            acts_tick = means + stds * noise
            # print("acts_tick", acts_tick.shape)
            # print("masked", acts_tick.shape)
            acts_tick = jnp.tanh(acts_tick)
            
            padded_acts = apply_padding_mask(acts_tick, masks)
            # jax.debug.print("pad {}\nmask {}\nact {}\n",padded_acts[0], masks[0], acts_tick[0])
            # jax.debug.print("masked {}\n{}", acts_tick[0], acts_tick_2[0])
            return padded_acts
        
        elif task == "gen_gmm":  
            
            def apply_padding_mask(expanded_array, padding_counts, pad_value=-2.0):
                x, n, z = expanded_array.shape
                
                # Create indices for each position in the batch dimension
                batch_indices = jnp.arange(n)
                
                # Reshape padding_counts and broadcast for comparison
                counts_expanded = padding_counts.reshape(x, 1)
                
                # Create a mask where True means we should keep the original value
                # and False means we should pad
                mask = batch_indices < counts_expanded
                
                # Expand the mask to match the full shape
                full_mask = jnp.broadcast_to(mask.reshape(x, n, 1), (x, n, z))
                
                # Apply the mask using where to conditionally select values
                result = jnp.where(full_mask, expanded_array, pad_value)
                
                return result
            
            action_dim = self.eval_env.unwrapped.action_dim
            batch_size = self.eval_env.unwrapped.max_batches
            
            # Split the policy output as specified
            means_all, log_stds_all, weight_logits = jnp.split(act_logits, 3, axis=-1)
           
            N = means_all.shape[0]
            # # Reshape all parameters to separate mixture components and action dimensions
            means_all = jnp.reshape(means_all, (N, 1, action_dim, self.sample_distribution)) 
            means_all = jnp.broadcast_to(means_all, (N, batch_size, action_dim, self.sample_distribution))
            log_stds_all = jnp.reshape(log_stds_all, (N, 1, action_dim, self.sample_distribution))
            log_stds_all = jnp.broadcast_to(log_stds_all, (N, batch_size, action_dim, self.sample_distribution))
            weight_logits = jnp.reshape(weight_logits, (N, 1, action_dim, self.sample_distribution))
            weight_logits = jnp.broadcast_to(weight_logits, (N, batch_size, action_dim, self.sample_distribution))
            
            # Clip log_stds to prevent numerical issues
            log_stds_all = jnp.clip(log_stds_all, -20, 2)
            stds_all = jnp.exp(log_stds_all)
            
            # print("weights_logits", weight_logits.shape, weight_logits)
            
            # Calculate weights from weight_logits (softmax across mixture components for each action dim)
            weights = jax.nn.softmax(weight_logits, axis=-1)  # shape: (N, action_dim, k)
            
           
            # Split random key for component selection and noise generation
            key_cat, key_noise = jax.random.split(random_key)
            
            
            # Sample mixture component indices for each action dimension separately
            # Initialize component_indices with the right shape
            component_indices = jnp.zeros((N, batch_size, action_dim), dtype=jnp.int32)
            
            indices = jax.random.categorical(key_cat, jnp.log(weights), axis=-1)
            indices = jnp.expand_dims(indices, axis=-1)

            # Gather the chosen means and stds based on component indices
            chosen_means = jnp.take_along_axis(means_all, indices, axis=-1)
            chosen_stds = jnp.take_along_axis(stds_all, indices, axis=-1)
            
            # Remove the last singleton dimension
            chosen_means = jnp.squeeze(chosen_means, axis=-1)  # shape: (N, batch_size, action_dim)
            chosen_stds = jnp.squeeze(chosen_stds, axis=-1)    # shape: (N, batch_size, action_dim)
            
            # Sample noise from standard normal distribution
            noise = jax.random.normal(key_noise, shape=chosen_means.shape)
            
            # Compute the final sampled actions
            acts_tick = chosen_means + chosen_stds * noise  # shape: (N, batch_size, action_dim)
            
            # Apply tanh to constrain actions to [-1, 1] range
            acts_tick = jnp.tanh(acts_tick)
            
            # Apply padding mask if necessary
            if masks is not None:
                acts_tick = apply_padding_mask(acts_tick, masks)
            
            return acts_tick
        
        elif task == "cor_gmm":
            def apply_padding_mask(expanded_array, padding_counts, pad_value=-2.0):
                x, n, z = expanded_array.shape
                
                # Create indices for each position in the batch dimension
                batch_indices = jnp.arange(n)
                
                # Reshape padding_counts and broadcast for comparison
                counts_expanded = padding_counts.reshape(x, 1)
                
                # Create a mask where True means we should keep the original value
                # and False means we should pad
                mask = batch_indices < counts_expanded
                
                # Expand the mask to match the full shape
                full_mask = jnp.broadcast_to(mask.reshape(x, n, 1), (x, n, z))
                
                # Apply the mask using where to conditionally select values
                result = jnp.where(full_mask, expanded_array, pad_value)
                
                return result
            
            action_dim = self.eval_env.unwrapped.action_dim
            batch_size = self.eval_env.unwrapped.max_batches
       
            N = act_logits.shape[0]
            weight_logits = act_logits[..., (2 * self.sample_distribution):]  
            weight_logits = jnp.reshape(weight_logits, (N, batch_size, action_dim, self.sample_distribution))
            
            
        
            
            gmm_params = act_logits[..., :(2 * self.sample_distribution)]  
            means, std = jnp.split(gmm_params, 2, axis=-1) 
            means_all = jnp.reshape(means, (N, 1, self.sample_distribution))  
            
            means_all = jnp.expand_dims(jnp.broadcast_to(means_all, (N, batch_size, self.sample_distribution)), axis=2)
            means_all = jnp.broadcast_to(means_all, (N, batch_size, action_dim, self.sample_distribution))
            log_stds_all = jnp.reshape(std, (N, 1, self.sample_distribution))  # shape: (N, action_dim, k)
            log_stds_all = jnp.expand_dims(jnp.broadcast_to(log_stds_all, (N, batch_size, self.sample_distribution)), axis=2)
            log_stds_all = jnp.broadcast_to(log_stds_all, (N, batch_size, action_dim, self.sample_distribution))
            # print("means", means_all[0], "std", log_stds_all[0])
            
            
            
         
            # Clip log_stds to prevent numerical issues
            log_stds_all = jnp.clip(log_stds_all, -20, 2)
            stds_all = jnp.exp(log_stds_all)
            
            # Calculate mixture weights (these are the same for all action dimensions)
            weights = jax.nn.softmax(weight_logits, axis=-1)  # shape: (N, k)
            
            # Get number of parallel environments
            N = means_all.shape[0]
            
            # Split random key for component selection and noise generation
            key_cat, key_noise = jax.random.split(random_key)
            
            # Sample mixture component indices based on weights
            # This selects which component to use for the entire action vector
            # Shape: (N,)
            component_indices = jax.random.categorical(key_cat, logits=weight_logits, axis=-1)
            component_indices = jnp.expand_dims(component_indices, axis=-1)
            # Gather the chosen means and stds based on component indices
            # We need to handle the indexing carefully to keep the action dimensions
            
        
           
            
            chosen_means = jnp.squeeze(jnp.take_along_axis(means_all, component_indices, axis=-1), axis=-1)
            chosen_stds = jnp.squeeze(jnp.take_along_axis(stds_all, component_indices, axis=-1), axis=-1)
            
            
        
            
            # Sample noise from standard normal distribution
            # Using the same key for all dimensions creates correlation
            noise = jax.random.normal(key_noise, shape=(N, batch_size, action_dim))
            
            # Compute the final sampled actions
            acts_tick = chosen_means + chosen_stds * noise  # shape: (N, batch_size, action_dim)
            
            # Apply tanh to constrain actions to [-1, 1] range
            acts_tick = jnp.tanh(acts_tick)
            
            # Apply padding mask if necessary
            if masks is not None:
                acts_tick = apply_padding_mask(acts_tick, masks)
            
            return acts_tick
        
        
        elif task == "full_params":  
            def apply_padding_mask(expanded_array, padding_counts, pad_value=-2.0):
                x, n, z = expanded_array.shape
                
                # Create indices for each position in the batch dimension
                batch_indices = jnp.arange(n)
                
                # Reshape padding_counts and broadcast for comparison
                counts_expanded = padding_counts.reshape(x, 1)
                
                # Create a mask where True means we should keep the original value
                # and False means we should pad
                mask = batch_indices < counts_expanded
                
                # Expand the mask to match the full shape
                full_mask = jnp.broadcast_to(mask.reshape(x, n, 1), (x, n, z))
                
                # Apply the mask using where to conditionally select values
                result = jnp.where(full_mask, expanded_array, pad_value)
                
                return result
            
            action_dim = self.eval_env.unwrapped.action_dim  # Ensure your environment defines action_dim
            batch_size = self.eval_env.unwrapped.max_batches

            # Split the policy output into means and log_stds.
            means, log_stds = jnp.split(act_logits, 2, axis=-1)
            
            N = means.shape[0]
            
            means = jnp.reshape(means, (N, batch_size, action_dim))
            log_stds = jnp.reshape(log_stds, (N, batch_size, action_dim))
            # print("means", means.shape, "log_stds", log_stds.shape)
            
            log_stds = jnp.clip(log_stds, -20, 2)
            stds = jnp.exp(log_stds)

            batch_size = self.eval_env.unwrapped.batch_size  # number of samples per environment

            # Get the number of parallel environments.
            

            # Broadcast parameters to match the batch size.
            # New shape becomes (N, batch_size, action_dim)
            # means = jnp.broadcast_to(means, (N, batch_size, action_dim))
            # stds = jnp.broadcast_to(stds, (N, batch_size, action_dim))
            # Sample noise from a standard normal distribution matching the shape.
            noise = jax.random.normal(random_key, shape=means.shape)

            # Compute the final sampled actions.
            acts_tick = means + stds * noise
            
            acts_tick = jnp.tanh(acts_tick)
            
            # Apply padding mask if necessary
            if masks is not None:
                acts_tick = apply_padding_mask(acts_tick, masks)

            
            return acts_tick
        
        elif task == "vae":  
            def apply_padding_mask(expanded_array, padding_counts, pad_value=-2.0):
                x, n, z = expanded_array.shape
                
                # Create indices for each position in the batch dimension
                batch_indices = jnp.arange(n)
                
                # Reshape padding_counts and broadcast for comparison
                counts_expanded = padding_counts.reshape(x, 1)
                
                # Create a mask where True means we should keep the original value
                # and False means we should pad
                mask = batch_indices < counts_expanded
                
                # Expand the mask to match the full shape
                full_mask = jnp.broadcast_to(mask.reshape(x, n, 1), (x, n, z))
                
                # Apply the mask using where to conditionally select values
                result = jnp.where(full_mask, expanded_array, pad_value)
                
                return result
            
            action_dim = self.eval_env.unwrapped.action_dim  # Ensure your environment defines action_dim
            batch_size = self.eval_env.unwrapped.max_batches

            # Split the policy output into means and log_stds.
            means, log_stds = jnp.split(act_logits, 2, axis=-1)
            
            N = means.shape[0]
            
            means = jnp.reshape(means, (N, batch_size, action_dim))
            log_stds = jnp.reshape(log_stds, (N, batch_size, action_dim))
            # print("means", means.shape, "log_stds", log_stds.shape)
            
            log_stds = jnp.clip(log_stds, -20, 2)
            stds = jnp.exp(log_stds)

            batch_size = self.eval_env.unwrapped.batch_size  # number of samples per environment

            # Get the number of parallel environments.
            

            # Broadcast parameters to match the batch size.
            # New shape becomes (N, batch_size, action_dim)
            # means = jnp.broadcast_to(means, (N, batch_size, action_dim))
            # stds = jnp.broadcast_to(stds, (N, batch_size, action_dim))
            # Sample noise from a standard normal distribution matching the shape.
            noise = jax.random.normal(random_key, shape=means.shape)

            # Compute the final sampled actions.
            acts_tick = means + stds * noise
            
            acts_tick = jnp.tanh(acts_tick)
            
            # Apply padding mask if necessary
            if masks is not None:
                acts_tick = apply_padding_mask(acts_tick, masks)

            
            return acts_tick
            

            
        return acts_tick

    
    def unroll_actors(self,random_key):
        """
        
        Starts with O_{tick},H_{tick-1} as the start state and 
            updates the state to O_(tick+rollout_len+1), H_(tick+rollout_len)
            Inheriting classes should optimize for timesteps tick to tick+rollout_len
        Returns:
            observations : list
            Numpy array of observations O_(tick) - O_(tick+rollout_len+1)
        actions : list
            Numpy array of actions from A_(tick) - A_(tick+rollout_len)
        rewards : list
           Numpy array of observations R_(tick) - R_(tick+rollout_len+1)
        terminations : list
            Numpy array of shape \gamma_(tick) - \gamma_(tick+rollout_len+1)
        critic_preds : list
            Numpy array of critic predictions V_(tick) - V_(tick+rollout_len+1)
        actor_preds : list
            Numpy array of actor predictions A_(tick) - A_(tick+rollout_len) and shape BXTXnum_actions
        hiddens: list
            A pytree of hidden states each leaf adds dimensions  num_envsXnum_seqs (num_seqs=rollout_len//sequence_length)
        hidden_indices: list
            A list of indices of the observations corresponding to the hidden states num_envs X num_seqs X seq_len 
            Indices contains indices for timestep tick to tick+rollout_len
        
        """
        #Unrolls the actor for rollout_len steps, takes rollout_len actions
        num_seqs=self.rollout_len//self.sequence_length
        h_tickminus1=self.h_tickminus1
        o_tick=self.o_tick
        r_tick=self.r_tick
        term_tick=self.term_tick
        actions=[]
        observations={}
        rewards=[]
        critic_preds=[]
        actor_preds=[]
        latent_means=[]
        latent_stds=[]
        terminations=[]
        hiddens=[] #We still store the hidden states for every start 
        hidden_indices=[] #To map the hidden states to the correct timestep
        infos=[]
        for t in range(self.rollout_len):
            #Add observation and reward and timestep tick
            # observations.append(o_tick.copy())
            # print("o_tick", type(o_tick), type(observations))
            observations = self.stack_dict_obs(o_tick, observations)
            # print(r_tick)
            rewards.append(r_tick.copy())
            terminations.append(term_tick.copy())
            random_key,model_key=jax.random.split(random_key)
            #Add hidden state for each sequence
            if t%self.sequence_length==0: #if it is time to update the hidden state
                #Store the hidden state
                hiddens.append(jax.tree_map(lambda x:x,h_tickminus1))
                hidden_indices.append(jnp.repeat(jnp.arange(t,t+self.sequence_length).reshape(1,-1),repeats=self.env.num_envs,axis=0))
            
           
            expanded_o = self.expand_o_tick(o_tick)
            
           
            # act_logits,v_tick,htick=self.actor_critic_fn(model_key,self.params,expanded_o,jnp.expand_dims(term_tick,1),#jnp.expand_dims(o_tick,1),jnp.expand_dims(term_tick,1),
            #                                              h_tickminus1)
            act_logits,v_tick,htick,latent_params=self.actor_critic_fn(model_key,self.params,expanded_o,jnp.expand_dims(term_tick,1),#jnp.expand_dims(o_tick,1),jnp.expand_dims(term_tick,1),
                                                         h_tickminus1)
            
            # print("expanded_o", expanded_o["mask"].shape)
            masks = expanded_o["mask"]
            
            
            
            
            # if self.use_gumbel_sampling and not self.continious_samlping:
            if self.use_gumbel_sampling and not self.continious_sampling:
                # sample action: Gumbel-softmax trick
                # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
                u = jax.random.uniform(random_key, shape=act_logits.shape)
                acts_tick=jnp.argmax(act_logits - jnp.log(-jnp.log(u)), axis=-1).squeeze(axis=-1)
            elif self.task == "batch":
                acts_tick = self.sampling_differ("batch", act_logits, random_key)
                # print(acts_tick.shape, "\n")
                acts_tick = jnp.squeeze(acts_tick)
                # acts_ticka = sampling_differ("sampling", act_logits, random_key)
                # print(acts_tick, "\n", acts_ticka, "\n")
            elif self.task == "sampling":
                acts_tick = self.sampling_differ("sampling", act_logits, random_key)
            elif self.task == "multibatch":
                # print("multibatch")
                acts_tick = self.sampling_differ("multibatch", act_logits, random_key)    
            elif self.task == "expanded_samp":
                # print("multibatch")
                acts_tick = self.sampling_differ("expanded_samp", act_logits, random_key)    
            elif self.task == "multidim":
                # print("multibatch")
                acts_tick = self.sampling_differ("multidim", act_logits, random_key)    
            elif self.task == "masked":
                # print("multibatch")
                acts_tick = self.sampling_differ("masked", act_logits, random_key, masks=masks)    
            elif self.task == "gen_gmm":
                # print("multibatch")
                acts_tick = self.sampling_differ("gen_gmm", act_logits, random_key, masks=masks)    
            elif self.task == "cor_gmm":
                # print("multibatch")
                acts_tick = self.sampling_differ("cor_gmm", act_logits, random_key, masks=masks)    
            elif self.task == "full_params":
                # print("multibatch")
                acts_tick = self.sampling_differ("full_params", act_logits, random_key, masks=masks) 
            elif self.task == "vae":
                # print("multibatch")
                acts_tick = self.sampling_differ("masked", act_logits, random_key, masks=masks)   
            
                
            
            
            
                
                
                
                
            else:
                acts_tick=jax.random.categorical(random_key,act_logits).squeeze(axis=-1)
            #Take a step in the environment
            
            # print(act_logits.shape, acts_tick.shape)
            # print("sampled action", acts_tick.shape, "policy_out", act_logits.shape)
            # jax.debug.print("acts_tick {} acts {}",acts_tick, act_logits)
           
            o_tickplus1,r_tickplus1,term_tickplus1,trunc_tickplus1,info=self.env.step(*jax_to_numpy(acts_tick))
            o_tickplus1,r_tickplus1=numpy_to_jax(o_tickplus1,r_tickplus1)
            term_tickplus1,trunc_tickplus1=numpy_to_jax(term_tickplus1,trunc_tickplus1,dtype=bool)
            term_tickplus1=jnp.logical_or(term_tickplus1,trunc_tickplus1)
            #Add action at timestep tick 
            critic_preds.append(v_tick.copy())
            actor_preds.append(act_logits.copy())
            # print("aa", len(actor_preds))
            actions.append(acts_tick.copy())
            infos.append(info)
            # print(act_logits.shape, "act_logits")
            # print(latent_params[0].shape, "latent_params")
            latent_means.append(latent_params[0])
            latent_stds.append(latent_params[1])
            o_tick=o_tickplus1
            r_tick=r_tickplus1
            h_tickminus1=htick
            term_tick=term_tickplus1
            self.tick+=1
        #add the last observation and reward
        
        # def stack_dict_obs(o_tick):
            
        
        # observations.append(o_tick)
        observations = self.stack_dict_obs(o_tick, observations)
        rewards.append(r_tick)
        terminations.append(term_tick)
        #get the value for timestep (tick+rollout_len+1), we need this to do bootstrapping
        random_key,model_key=jax.random.split(random_key)
        expanded_o = self.expand_o_tick(o_tick)
        _,v_tick,_,latent=self.actor_critic_fn(model_key,self.params,expanded_o,jnp.expand_dims(term_tick,1),h_tickminus1)#jnp.expand_dims(o_tick,1),jnp.expand_dims(term_tick,1),h_tickminus1)
        critic_preds.append(v_tick)
        #Update to timestep
        self.o_tick=o_tick.copy()
        self.r_tick=r_tick.copy()
        self.h_tickminus1=jax.tree_map(lambda x:x,h_tickminus1) #Copy the hidden state, it can be arbitrary tree structure
        self.term_tick=term_tick.copy()
        #Shape is num_actorsXrollout_lenX*...
        hidden_stacked=jax.tree_map(lambda *args: jnp.stack(args,1), *hiddens)
        
        
       
        return Namespace(**{
            # 'observations':jnp.stack(observations,1),
            'observations': observations,
            'actions':jnp.stack(actions,1),
            'rewards':jnp.stack(rewards,1),
            'latent_means':jnp.stack(latent_means,1),
            'latent_stds':jnp.stack(latent_stds,1),
            'terminations':jnp.stack(terminations,1),
            'infos':infos,
            'critic_preds':jnp.squeeze(jnp.stack(critic_preds,1),axis=-1), #During unrolling phase, we don't need the time dimension as it is one, instead we need the rollout_len dimension
            'actor_preds':jnp.stack(actor_preds,1),
            'hiddens':hidden_stacked,
            'hidden_indices':jnp.stack(hidden_indices,1)
        })
        

    def evaluate(self,random_key,eval_episodes):
        #Evaluate the agent for evaluation_steps
        #Create a single zero hidden state
        #Get the hidden state from the first actor
        o_tick,_=self.eval_env.reset()
        episode_lens=[]
        episode_avgreturns=[]
        rollouts=[]
        term_tick=jnp.zeros((1,1),dtype=bool)  #Initialize terminal state to False
        #Initialize zero hidden state at the start of each episode, shape is infered from the hidden state of the first environment
        h_tickminus1=jax.tree_map(lambda x:jnp.expand_dims(jnp.zeros(x[0].shape),0) ,self.h_tickminus1)
        # print("eps", eval_episodes)
        for i in tqdm.tqdm(range(eval_episodes)):
            done=False
            rewards=[]
            
            while not done:
                #Take a step in the environment
                random_key,model_key=jax.random.split(random_key)
                # for k in o_tick.keys():
                #     print("bef", k, o_tick[k].shape)
               
                expanded_o = self.expand_o_tick(o_tick, eval=True)
                expanded_o = self.expand_o_tick(expanded_o, eval=True)
                # for k in expanded_o.keys():
                #     print("aft", k, expanded_o[k].shape)
                
                act_logits,v_tick,htick,latent=self.actor_critic_fn(model_key,self.params,expanded_o,term_tick,h_tickminus1)
                # if hasattr(self,'arg_max') and self.arg_max:
                #     acts_tick=jnp.argmax(act_logits,axis=-1)
                # else:
                #     if self.use_gumbel_sampling:
                #          # sample action: Gumbel-softmax trick
                #         # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
                #         u = jax.random.uniform(random_key, shape=act_logits.shape)
                #         acts_tick=jnp.argmax(act_logits - jnp.log(-jnp.log(u)), axis=-1).squeeze(axis=-1)
                #     else:
                #         acts_tick=jax.random.categorical(random_key,act_logits).squeeze(axis=-1)
                acts_tick = self.sampling_differ(self.task, act_logits,random_key, masks=expanded_o["mask"])
                
                # print("the shapes", act_logits.shape, acts_tick.shape)
                o_tick,r_tick,term,trunc,info=self.eval_env.step(*jax_to_numpy(acts_tick))
                # print("o_tick", o_tick, "r_tick", r_tick, "term", term, "trunc", trunc, "info", info)
                o_tick,r_tick=numpy_to_jax(o_tick,r_tick)
                done=term or trunc
                term_tick=jnp.array([[done]],dtype=bool) #Carry forward the termination signal for the next timestep, shape expected by the actor_critic_fn is BXT
                rewards.append(r_tick)
                h_tickminus1=htick
            #Get the rollout frames
            # print("info", jnp.array(info["final_info"]["actions"]).shape) 
            # print("info", info["final_info"]["eval_scaled_diff"].shape) 
            actions = jnp.array(info["final_info"]["actions"])
            scaled_diff = jnp.array(info["final_info"]["eval_scaled_diff"])
            rew = jnp.array(info["final_info"]["rewards"])
            
            
            max_x = jnp.array(info["final_info"]["max_x"])
            # print("max_cof", max_x.shape, max_y.shape)
            conc_max = jnp.concatenate([max_x, jnp.array([[0,0]])], axis=1)
            # print("max_cof", conc_max.shape)
            
            eval_rew = jnp.zeros((rew.shape[0] * actions.shape[1],1), dtype=jnp.float32)  # Create an array filled with zeros
            eval_rew = eval_rew.at[jnp.arange(rew.shape[0]) * actions.shape[1],1].set(rew)
            # print("rew", rew)
            # print(eval_rew)
            # print("scaled_diff", scaled_diff.shape, rew.shape, actions.shape)
         
            actions = jnp.reshape(actions, (actions.shape[0] * actions.shape[1], actions.shape[2]))
            scaled_diff = jnp.reshape(scaled_diff, (scaled_diff.shape[0] * scaled_diff.shape[1], 1))
            
            # print("actions", actions.shape)
            combined = jnp.concatenate([actions, scaled_diff, eval_rew], axis=1)
            table = jnp.concatenate([conc_max, combined], axis=0)
            
            rollouts = table
            episode_lens.append(len(rewards))
            rewards=jnp.array(rewards,dtype=jnp.float32)
            avg_return=rlax.discounted_returns(rewards,self.gamma*jnp.ones_like(rewards),jnp.zeros_like(rewards)).mean()
            episode_avgreturns.append(avg_return)
        avg_episode_len=jnp.array(episode_lens).mean()
        avg_episode_return=jnp.array(episode_avgreturns).mean()
        return avg_episode_len,avg_episode_return,rollouts