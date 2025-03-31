import jax, jax.numpy as jnp
from src.agents.ppo_dic_inherits.inh_agents.standard_sampling import SamplingImplBase

class FullParamsSampling(SamplingImplBase):
    
    def sampling_differ(self, act_logits, key, masks):
        means, log_stds = jnp.split(act_logits,2,-1)
        N = means.shape[0] 
        means = means.reshape((N,self.batch_size,self.action_dim))
        log_stds = jnp.clip(log_stds.reshape((N,self.batch_size,self.action_dim)), -20,2)
        noise = jax.random.normal(key, means.shape)
        
        
        out = jnp.tanh(means + jnp.exp(log_stds)*noise)
        return self.apply_padding_mask(out, masks)
    
    def gaussian_log_prob(self, actions, act_logits):
        epsilon = 1e-6
        
        act_logits = jnp.reshape(act_logits, (act_logits.shape[0],
                                                act_logits.shape[1],
                                                1,
                                                act_logits.shape[-1]))
        
                
        # Split into means and log_stds; each will be shape (N, T, 1, action_dim)
        means, log_stds = jnp.split(act_logits, 2, axis=-1)
        log_stds = jnp.clip(log_stds, -20, 2)
        
        # jax.debug.print("means:\n{}\n", means[0,0])
        
        stds = jnp.exp(log_stds)
        
        N = means.shape[0]
        S = means.shape[1]
        
        # Broadcast means, stds, log_stds from sample dimension (1) to batch_size.
        means = jnp.reshape(means, (N, S, self.batch_size, self.action_dim))
        stds = jnp.reshape(stds, (N, S, self.batch_size, self.action_dim))
        log_stds = jnp.reshape(log_stds, (N, S, self.batch_size, self.action_dim))
        
                                            
        # Unsquash the actions: a = tanh(u)  => u = atanh(a)
        # Ensure actions are in (-1+epsilon, 1-epsilon)
        u = jnp.arctanh(jnp.clip(actions, -1 + epsilon, 1 - epsilon))
        
        # Compute the Gaussian log-probability for u under N(means, stds):
        # log N(u | mean, std) = -0.5 * ( ((u - mean)^2 / std^2) + 2*log_std + log(2*pi) )
        log_prob_per_dim = -0.5 * ( ((u - means)**2 / (stds**2)) 
                                    + 2.0 * log_stds 
                                    + jnp.log(2 * jnp.pi) )
        # Sum the log-probabilities over the action dimensions.
        base_log_prob = jnp.sum(log_prob_per_dim, axis=-1)  # shape (N, T, batch_size)
        
        # Tanh correction: a = tanh(u) => derivative: 1 - tanh(u)^2 = 1 - a^2
        # So add correction term: sum(log(1 - a^2 + epsilon)) over dims.
        correction = jnp.sum(jnp.log(1 - actions**2 + epsilon), axis=-1)  # shape (N, T, batch_size)
        
        log_prob = base_log_prob - correction
        
        # Optionally, mask invalid actions (if the first element of action is -2, mark log_prob 0)
        valid_mask = (actions[..., 0] != -2)
        log_prob = jnp.where(valid_mask, log_prob, 0.0)
        log_prob = jnp.sum(log_prob, axis=-1)  # shape (N, T)
        
        return log_prob

    def entropy(self, logits, mask):
       
        logits = jnp.reshape(logits, (logits.shape[0],
                                  logits.shape[1],
                                  1,
                                  logits.shape[-1]))
    
        # Split into means and log_stds; each becomes shape (N, T, 1, action_dim)
        _, log_stds = jnp.split(logits, 2, axis=-1)
        
        # Broadcast the means and log_stds to shape (N, T, batch_size, action_dim)
        log_stds = jnp.reshape(log_stds, (log_stds.shape[0], log_stds.shape[1], self.batch_size, self.action_dim))
        
        # Clip log_stds for numerical stability
        log_stds = jnp.clip(log_stds, -20, 2)
        
        # Compute the entropy per action dimension using the standard Gaussian entropy formula:
        # H = 0.5 * (log(2*pi*e) + 2*log_std)
        entropy_per_dim = 0.5 * (jnp.log(2 * jnp.pi * jnp.e) + 2 * log_stds)
        
        # Sum over the action dimension to get shape (N, T, batch_size)
        ent = jnp.sum(entropy_per_dim, axis=-1)
        
        # Broadcast the mask (of shape (N, T, 1)) to (N, T, batch_size)
        mask = jnp.squeeze(mask, axis=-1)
        mask = jnp.broadcast_to(mask, (mask.shape[0], mask.shape[1], self.batch_size))

        # Zero out entropy for invalid (masked) entries
        ent = jnp.where(mask, ent, 0.0)
        ent = jnp.sum(ent, axis=-1)
        
        return ent