import jax
import jax.numpy as jnp
import equinox as eqx # Flowjax flows are often eqx.Modules

# Assuming your import path is correct for your project structure
from src.agents.ppo_dic_inherits.inh_agents.standard_sampling_jax import SamplingImplBaseJax

from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.bijections import Tanh, Chain, AbstractBijection



class NormFlowAgent(SamplingImplBaseJax):
    """
    A sampling implementation using a Multivariate Normal base distribution,
    transformed by a provided flowjax flow object (passed during method calls),
    and finally by a Tanh bijection.

    The parameters of the `flow_object` are managed externally (e.g., in AgentTrainState).
    """
    # No flow_module stored here anymore

    def __init__(self, 
                 max_batch: int, 
                 action_dim: int, 
                 sampling_distribution: int = 1):
        """
        Initializes the FlowMVNJax sampler.

        Args:
            max_batch: Maximum batch size for sampling (number of samples drawn per distribution).
            action_dim: Dimensionality of the action space.
            sampling_distribution: Passed to superclass.
        """
        super().__init__(max_batch, action_dim, sampling_distribution)
        self.cov_dim = self.action_dim
        # Store action_dim if needed by flow structure creation later, but
        # typically the flow object passed in will already be configured.
        # self.action_dim = action_dim 

    def generate_mvn_params(self, act_logits):
        """
        act_logits: shape (dim + dim*(dim+1)//2,)
        Returns: loc (action_dim,), covariance (action_dim, action_dim)
        """
        
        loc = act_logits[:self.cov_dim]
        tril_values = act_logits[self.cov_dim:]

        tril_indices = jnp.tril_indices(self.cov_dim)
        L = jnp.zeros((self.cov_dim, self.cov_dim)).at[tril_indices].set(tril_values)

        diag_indices = jnp.diag_indices(self.cov_dim)
        L = L.at[diag_indices].set(L[diag_indices])

        covariance = L @ L.T
        
        # loc = jnp.full((self.cov_dim,), 8)  
        # covariance = jnp.eye(self.cov_dim) * 8  # Ensure positive definiteness
      
        
        return loc, covariance

    def _get_final_bijection(self, flow_object: AbstractBijection | None):
        """
        Constructs the final bijection chain: Optional Flow -> Tanh.

        Args:
            flow_object: The flowjax flow instance (e.g., BlockNeuralAutoregressiveFlow)
                         containing its current parameters, or None.

        Returns:
            A flowjax.bijections.AbstractBijection object representing the combined transform.
        """
        tanh_bijection = Tanh(shape=(self.cov_dim,)) 
        

        if flow_object is not None:
            # Chain the bijections: Input -> flow_object -> Tanh
            # Order matters for Chain: applies first element, then second, etc
            final_bijection = Chain([flow_object, tanh_bijection])
        else:
            # No additional flow module, just Tanh
            #in current setting this is always the case
            final_bijection = tanh_bijection
            
        return final_bijection

    def sampling_differ(self, act_logits_one_dist, flow_object: AbstractBijection | None, key, mask_threshold_array):
        """
        Samples actions for a single distribution defined by act_logits_one_dist,
        transformed by the provided flow_object.
        Intended to be vmapped if batch processing over multiple distributions is needed.

        Args:
            act_logits_one_dist: Parameters for the MVN base distribution (1D array).
            flow_object: The flow instance (e.g., BNAF) with current parameters, or None.
            key: JAX PRNGKey for sampling.
            mask_threshold_array: A JAX array like `jnp.array([count])` for masking.

        Returns:
            JAX array of shape (self.batch_size, action_dim) containing samples, padded.
        """
        loc, cov = self.generate_mvn_params(act_logits_one_dist)
        base_dist = MultivariateNormal(loc=loc, covariance=cov)
        final_bijection = self._get_final_bijection(flow_object)
        
        # Create the final transformed distribution
        final_dist = Transformed(base_dist, final_bijection)

        # Sample `self.batch_size` times from this single distribution definition.
        samples = final_dist.sample(key, sample_shape=(self.batch_size,)) 
        
        # Apply padding mask
        masked_samples = self.apply_padding_mask(samples, mask_threshold_array)
        return masked_samples


    
    def gaussian_log_prob(self, actions, act_logits, flow_object: AbstractBijection | None):
        epsilon = 1e-6
        
        N = act_logits.shape[0]
       
       
        get_params = jax.vmap(self.generate_mvn_params)
        
        locs, covs = get_params(act_logits)  # locs: (N, T, A), covs: (N, T, A, A)
        
        def log_prob_fn(loc, cov, action):
            base_dist = MultivariateNormal(loc=loc, covariance=cov)
            final_bijection = self._get_final_bijection(flow_object)
            
            # Create the final transformed distribution
            final_dist = Transformed(base_dist, final_bijection)
            log_prob = final_dist.log_prob(action)
            return log_prob
        
        log_prob_vmap = jax.vmap(log_prob_fn)
        
        u = jnp.clip(actions, -1 + epsilon, 1 - epsilon)

        log_probs = log_prob_vmap(locs, covs, u)  # shape: (N, T, B)
        
        valid_mask = (actions[..., 0] != -2)
        log_prob = jnp.where(valid_mask, log_probs, 0.0)
        log_prob = jnp.sum(log_prob, axis=-1)  # shape (N, T)
        
        
        
        # jax.debug.print("Log probability shape: {}, std {}", log_prob[0,0], act_logits[0,0])

        return log_prob
        
        # jax.debug.print("acts shape: {} {}", actions[0,0], act_logits[0,0])

        

    
    
    def entropy(self, logits, mask, flow_object: AbstractBijection | None, key, num_samples=20):
        epsilon = 1e-6
      
        
        N, D = logits.shape
    
        # Vectorized function to compute entropy for a single (N, T) pair
        def compute_entropy_single(logit, subkey):
            loc, cov = self.generate_mvn_params(logit)
            base_dist = MultivariateNormal(loc, cov)
            final_bijection = self._get_final_bijection(flow_object)
            final_dist = Transformed(base_dist, final_bijection)

            # Generate samples and compute log-probabilities
            sample_keys = jax.random.split(subkey, num_samples)
            samples = jax.vmap(lambda k: final_dist.sample(k))(sample_keys)
            u = jnp.clip(samples, -1 + epsilon, 1 - epsilon)
            log_probs = jax.vmap(final_dist.log_prob)(u)
            
            # jax.debug.print("sample shape: {} {} {}", u[0], log_probs[0], cov)

            # Estimate entropy
            
            return -jnp.mean(log_probs)
        
        logits_flat = logits.reshape(-1, D)
        total = logits_flat.shape[0]

        # Generate unique keys for each (N, T) pair
        keys = jax.random.split(key, total)

        # Vectorized entropy computation
        summed_entropy = jax.vmap(compute_entropy_single)(logits_flat, keys)
        
        # jax.debug.print("summed_entropy shape: {}", summed_entropy.shape)
        
        # Reshape back to (N, T)
        # return entropy_estimates.reshape(N, T)
        
       
        # summed_entropy = jnp.sum(entropy_estimates, axis=-1) # Shape: (N, T, batch_size)
        
        og_mask = jnp.reshape(mask, (mask.shape[0], mask.shape[1]))  # Remove last dimension if it's 1
        B = summed_entropy.shape[-1]
        batch_idx = jnp.arange(B)  # (B,)

        valid_mask = batch_idx < og_mask[..., None]  # (N, T, B) ← broadcasts correctly
        masked_entropy = jnp.where(valid_mask, summed_entropy, 0.0)  # (N, T, B)
        
        
       
        final_entropy = jnp.sum(masked_entropy, axis=-1) / og_mask
        # jax.debug.print("correction shape: {} base {} final {}", correction[0,0], base_entropy[0,0], final_entropy[0,0])
        
        
      
        
        return final_entropy

