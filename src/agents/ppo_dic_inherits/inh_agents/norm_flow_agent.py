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
        
        #Act logist of shape (dim + dim*(dim+1)//2,)
        #Mask threshold array of shape (1,)
        # print("sampling_differ", act_logits_one_dist.shape, flow_object, key.shape, mask_threshold_array.shape)
        # Example output for act_dim =2, batch size = 3
        # sampling_differ (5,) None () (1,)
        
        loc, cov = self.generate_mvn_params(act_logits_one_dist)
        base_dist = MultivariateNormal(loc=loc, covariance=cov)
        final_bijection = self._get_final_bijection(flow_object)
        
        # Create the final transformed distribution
        final_dist = Transformed(base_dist, final_bijection)

        # Sample `self.batch_size` times from this single distribution definition.
        samples = final_dist.sample(key, sample_shape=(self.batch_size,)) 
        
        # Apply padding mask
        masked_samples = self.apply_padding_mask(samples, mask_threshold_array)
        
        
        # print("sample out", masked_samples.shape)
        # Example output for act_dim =2, batch size = 3
        #sample out (3, 2)
        
        
        return masked_samples


    
    def gaussian_log_prob(self, actions, act_logits, flow_object: AbstractBijection | None):
        epsilon = 1e-6
        
        # print("gaussian_log_prob", actions.shape, act_logits.shape)
        # Example output for act_dim =2, batch size = 3
        # sampling_differ (64,3,2) (64,5)
        
        N = act_logits.shape[0]
       
        # print("act_logits shape", act_logits.shape, "actions shape", actions.shape)
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
        
        
        
        # print("log_prob shape", log_prob.shape)
        # Example output for act_dim =2, batch size = 3, rollout length = 64
        #gaussian_log_prob (64, 3, 2) (64, 5)
        

        return log_prob
        

        

    
    
    def entropy(self, logits, mask, flow_object: AbstractBijection | None, key, num_samples=20):
        # print("entropy", logits.shape, mask.shape, key.shape)
        #Example output for act_dim =2, batch size = 3, rollout length = 64
        #entropy (64, 5) (64, 1) ()
        epsilon = 1e-6
      
        # print("entropy", logits.shape, mask.shape, key.shape)
        
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
            
          
            
            return -jnp.mean(log_probs)
        
        logits_flat = logits.reshape(-1, D)
        total = logits_flat.shape[0]

        # Generate unique keys for each (N, T) pair
        keys = jax.random.split(key, total)

        # Vectorized entropy computation
        summed_entropy = jax.vmap(compute_entropy_single)(logits_flat, keys)
        
      
        og_mask = jnp.reshape(mask, (mask.shape[0], mask.shape[1]))  # Remove last dimension if it's 1
        B = summed_entropy.shape[-1]
        batch_idx = jnp.arange(B)  # (B,)

        valid_mask = batch_idx < og_mask[..., None]  # (N, T, B) ← broadcasts correctly
        masked_entropy = jnp.where(valid_mask, summed_entropy, 0.0)  # (N, T, B)
        
        
       
        final_entropy = jnp.sum(masked_entropy, axis=-1) / og_mask
        
        
        # print("final_entropy shape", final_entropy.shape)
        #Example output for act_dim =2, batch size = 3, rollout length = 64
        #final_entropy shape (64, 1)
        
        
        return final_entropy
    
    
    # def get_pdf(self, act_logits_one_dist, flow_object: AbstractBijection | None, x_values: jnp.ndarray) -> jnp.ndarray:
    #     print("get_pdf", act_logits_one_dist.shape, flow_object, x_values.shape)
    #     # Example output for act_dim =1, amount of points = 200
    #     #get_pdf (2,) None (200, 1)
    #     """
    #     Computes the probability density function (PDF) of the policy for given x_values.
    #     x_values are assumed to be in the normalized space [-1, 1].

    #     Args:
    #         act_logits_one_dist: Parameters for the MVN base distribution for a single policy.
    #         flow_object: The flow instance (e.g., BNAF) with current parameters, or None.
    #         x_values: JAX array of shape (num_points, action_dim) at which to evaluate the PDF.
    #                   These values should be in the range [-1, 1].

    #     Returns:
    #         JAX array of shape (num_points,) containing PDF values.
    #     """
    #     loc, cov = self.generate_mvn_params(act_logits_one_dist)
    #     base_dist = MultivariateNormal(loc=loc, covariance=cov)
    #     final_bijection = self._get_final_bijection(flow_object)
        
    #     final_dist = Transformed(base_dist, final_bijection)

    #     # log_prob will compute log p(x) = log p_z(f^{-1}(x)) + log |det J_{f^{-1}}(x)|
    #     # where f is final_bijection.
    #     # x_values are in the target space of final_bijection (i.e., after Tanh, so in [-1, 1])
    #     log_probs = final_dist.log_prob(x_values)
        
    #     print("pdf shape", log_probs.shape)
    #     # Example output for act_dim =1, amount of points = 200
    #     #pdf shape (200,)
        
    #     return jnp.exp(log_probs)
    
    def get_pdf(self, act_logits_one_dist, flow_object: AbstractBijection | None, x_values: jnp.ndarray) -> jnp.ndarray:
        # print("get_pdf", act_logits_one_dist.shape, flow_object, x_values.shape)
        # Example output for act_dim =1, amount of points = 200
        #get_pdf (2,) None (200, 1)
        """
        Computes the probability density function (PDF) of the policy for given x_values.
        x_values are assumed to be in the normalized space [-1, 1].

        Args:
            act_logits_one_dist: Parameters for the MVN base distribution for a single policy.
            flow_object: The flow instance (e.g., BNAF) with current parameters, or None.
            x_values: JAX array of shape (num_points, action_dim) at which to evaluate the PDF.
                      These values should be in the range [-1, 1].

        Returns:
            JAX array of shape (num_points,) containing PDF values.
        """
        loc, cov = self.generate_mvn_params(act_logits_one_dist)
        base_dist = MultivariateNormal(loc=loc, covariance=cov)
        final_bijection = self._get_final_bijection(flow_object)
        
        final_dist = Transformed(base_dist, final_bijection)

        # log_prob will compute log p(x) = log p_z(f^{-1}(x)) + log |det J_{f^{-1}}(x)|
        # where f is final_bijection.
        # x_values are in the target space of final_bijection (i.e., after Tanh, so in [-1, 1])
        log_probs = final_dist.log_prob(x_values)
        
        # print("pdf shape", log_probs.shape)
        # Example output for act_dim =1, amount of points = 200
        #pdf shape (200,)
        
        return jnp.exp(log_probs)

