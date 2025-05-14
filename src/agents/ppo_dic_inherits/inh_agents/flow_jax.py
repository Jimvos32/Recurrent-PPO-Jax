import jax, jax.numpy as jnp
from src.agents.ppo_dic_inherits.inh_agents.standard_sampling import SamplingImplBase
from src.agents.ppo_dic_inherits.inh_agents.standard_sampling_jax import SamplingImplBaseJax
from flowjax.distributions import MultivariateNormal, Transformed
from flowjax.bijections import Tanh
import time

class FlowMVNJax(SamplingImplBaseJax):
    def __init__(self, max_batch, action_dim, sampling_distribution=1):
        super().__init__(max_batch, action_dim, sampling_distribution)
        #self.cov_dim = self.action_dim * self.batch_size #For a complete joint action
        self.cov_dim = self.action_dim
        
    def sampling_differ(self, act_logits, key, masks):
        #logits of shape (N, 1, (dim + dim * (dim + 1) // 2))
        
        print("sampling_differ", act_logits.shape, key.shape, masks.shape)
       
      
        loc, cov = self.generate_mvn_params(act_logits)

     
        mvn = MultivariateNormal(loc=loc, covariance=cov)
        tanh_dist = Transformed(mvn, Tanh(shape=(self.cov_dim,)))
        sample = tanh_dist.sample(key, sample_shape=(self.batch_size,)) 
             
        masked = self.apply_padding_mask(sample, masks)  # shape: (N, B, A)
        # jax.debug.print("masked shape {} {}", masked[0,0], samples[0,0])
        return masked
    
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
        

    
    def gaussian_log_prob(self, actions, act_logits):
        epsilon = 1e-6
        
        N = act_logits.shape[0]
       
        # Get loc + cov from logits: shape (N, T, D)
        get_params = jax.vmap(self.generate_mvn_params)
        
        print("gaussian_log_prob", actions.shape, act_logits.shape)
        locs, covs = get_params(act_logits)  # locs: (N, T, A), covs: (N, T, A, A)
        # jax.debug.print("locs shape: {} {}", locs[0,0], covs[0,0])
        print("gaussian_log_prob", actions.shape, act_logits.shape, locs.shape, covs.shape) 
        
        # jax.debug.print("acts shape: {} {}", actions[0,0], act_logits[0,0])

        def log_prob_fn(loc, cov, action):
            mvn = MultivariateNormal(loc=loc, covariance=cov)
            tanh_dist = Transformed(mvn, Tanh(shape=(self.cov_dim,)))
            ret = tanh_dist.log_prob(action)
            # jax.debug.print("act_logits shape: {} {} {}", cov, action, ret)
            return ret  # shape: (B,)

        # vmapped over N, T
        log_prob_vmap = jax.vmap(log_prob_fn)
        
        u = jnp.clip(actions, -1 + epsilon, 1 - epsilon)

        log_probs = log_prob_vmap(locs, covs, u)  # shape: (N, T, B)
        
        valid_mask = (actions[..., 0] != -2)
        log_prob = jnp.where(valid_mask, log_probs, 0.0)
        log_prob = jnp.sum(log_prob, axis=-1)  # shape (N, T)
        
        
        
        # jax.debug.print("Log probability shape: {}, std {}", log_prob[0,0], act_logits[0,0])

        return log_prob
        
        
        
        
    def entropy(self, logits, mask, key=None, num_samples=20):
        #logits shape (N, T, 1, dim + dim * (dim + 1) // 2) 
        epsilon = 1e-6
      
        
        N, D = logits.shape
        
        print("entropy", logits.shape, mask.shape, key.shape)
        

        # Vectorized function to compute entropy for a single (N, T) pair
        def compute_entropy_single(logit, subkey):
            loc, cov = self.generate_mvn_params(logit)
            base_dist = MultivariateNormal(loc, cov)
            tanh_dist = Transformed(base_dist, Tanh(shape=(self.cov_dim,)))

            # Generate samples and compute log-probabilities
            sample_keys = jax.random.split(subkey, num_samples)
            samples = jax.vmap(lambda k: tanh_dist.sample(k))(sample_keys)
            u = jnp.clip(samples, -1 + epsilon, 1 - epsilon)
            log_probs = jax.vmap(tanh_dist.log_prob)(u)
            
            # jax.debug.print("sample shape: {} {} {}", u[0], log_probs[0], cov)

            # Estimate entropy
            
            return -jnp.mean(log_probs)

        # Flatten logits for mapping
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