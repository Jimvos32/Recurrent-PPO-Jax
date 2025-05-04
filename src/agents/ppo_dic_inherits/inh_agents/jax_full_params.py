import jax, jax.numpy as jnp
from src.agents.ppo_dic_inherits.inh_agents.standard_sampling_jax import SamplingImplBaseJax

class FullParamsSamplingJax(SamplingImplBaseJax):
    
    def sampling_differ(self, act_logits, key, masks):
        
        
      
        
        means, scale = jnp.split(act_logits,2,-1)
        
      
        means = means.reshape((self.batch_size,self.action_dim))
        scale = scale.reshape((self.batch_size,self.action_dim))
        
        # self.ent_schedule(update_tick) 
        noise = jax.random.normal(key, means.shape)
        
        out = means + scale * noise
        
        # scale = jnp.full_like(scale, 0.1)
        # means = jnp.full_like(means, 0.0)
        
        out = jnp.tanh(means + scale * noise)
        
        # print("out shape", out.shape, means.shape, scale.shape, noise.shape)
        
        # jax.debug.print("mean {}, std {} noise {} out {}\n", jnp.mean(means, axis=0), jnp.mean(scale[0,0], axis=0), jnp.mean(noise[0,0], axis=0), jnp.mean(out, axis=0))
        # jax.debug.print("lean {}, ltd {} loise {} lut {}\n",  means[0,0], scale[0,0], noise[0,0], out[0,0])
        
        
        x = self.apply_padding_mask(out, masks)
        
        # print("padding mask shape", x.shape, "out shape", out.shape, "masks shape", masks.shape)
        
        # jax.debug.print("out {}\n masked {}\nmasks {}", out, x, masks)
        return x
    
    def gaussian_log_prob(self, actions, act_logits):
        epsilon = 1e-6
        
        # print("gaussian_log_prob", actions.shape, act_logits.shape)
        
        # act_logits = jnp.reshape(act_logits, (act_logits.shape[0],
        #                                         act_logits.shape[1],
        #                                         1,
        #                                         act_logits.shape[-1]))
        
                
        # Split into means and log_stds; each will be shape (N, T, 1, action_dim)
        N = act_logits.shape[0]
        means, std_out = jnp.split(act_logits, 2, axis=-1)
        
        
     
        # Broadcast means, stds, log_stds from sample dimension (1) to batch_size.
        means = jnp.reshape(means, (N, self.batch_size, self.action_dim))
        stds = jnp.reshape(std_out, (N, self.batch_size, self.action_dim))
        
                                            
        # Unsquash the actions: a = tanh(u)  => u = atanh(a)
        # Ensure actions are in (-1+epsilon, 1-epsilon)
        u = jnp.arctanh(jnp.clip(actions, -1 + epsilon, 1 - epsilon))
        
        
        
        log_prob_per_dim = jax.scipy.stats.norm.logpdf(u, means, stds)
    
        # Sum log probabilities across action dimensions
        
        # print("log_prob_per_dim shape", log_prob_per_dim.shape, "means shape", means.shape, "stds shape", stds.shape)
        base_log_prob = jnp.sum(log_prob_per_dim, axis=-1)  # Shape: (N, T, batch_size)
        # print("base_log_prob shape", base_log_prob.shape)
        
        # Calculate the log determinant of Jacobian for the tanh transformation
        # log|det(d/du tanh(u))| = log(1 - tanh^2(u)) = log(1 - actions^2)
        # Using numerically stable version: 2 * (log(2) - u - softplus(-2*u))
        log_det_jacobian = jnp.sum(2 * (jnp.log(2) - u - jax.nn.softplus(-2 * u)), axis=-1)
        
        # Subtract the log determinant to get corrected log probability
        log_prob = base_log_prob - log_det_jacobian  # Shape: (N, T, batch_size)
        
        
        
        # Optionally, mask invalid actions (if the first element of action is -2, mark log_prob 0)
        valid_mask = (actions[..., 0] != -2)
        
        
        
        # print("valid_mask shape", valid_mask.shape, "actions shape", actions.shape)
        log_prob = jnp.where(valid_mask, log_prob, 0.0)
        # jax.debug.print("valid_mask shape: \n{} actions shape: \n{} logprob\n{}", valid_mask, actions, log_prob)
        log_prob = jnp.sum(log_prob, axis=-1)  # shape (N, T)
        
        # jax.debug.print("Log probability shape: {}, std {} mean {}", log_prob[0,0], stds[0,0], means[0,0])
        # print("Log probability shape", log_prob.shape, stds.shape, means.shape)
        
        print("log_prob out  shape", log_prob.shape, actions.shape, act_logits.shape)
        
        
        return log_prob

    def entropy(self, logits, mask, key=None):#tanh in entropy calculation
        
        N = logits.shape[0]
        # logits = jnp.reshape(logits, (N, logits.shape[-1]))
        
        means, stds = jnp.split(logits, 2, axis=-1)
        
       
        # New shape: (N, T, batch_size, action_dim)
        means = jnp.reshape(means, (N, self.batch_size, self.action_dim))
        stds = jnp.reshape(stds, (N, self.batch_size, self.action_dim))
        
     
        # means = jnp.full_like(means, 0.0)
        # stds = jnp.full_like(stds, 0.1)
       
        base_entropy_per_sample = 0.5 * (jnp.log(2 * jnp.pi * jnp.e) + 2 * jnp.log(stds))
        
        base_entropy = base_entropy_per_sample
        
    
        n_quantiles = 100
        qs = jnp.linspace(0.01, 0.99, n_quantiles)
        
      
        std_normal_quantiles = jax.scipy.stats.norm.ppf(qs) 
        
   
        means_expanded = jnp.expand_dims(means, axis=-1)    # Shape: (N, T, batch_size, action_dim, 1)
        stds_expanded = jnp.expand_dims(stds, axis=-1)      # Shape: (N, T, batch_size, action_dim, 1)
        quantiles_expanded = jnp.reshape(std_normal_quantiles, (1, 1, 1, n_quantiles))
        
        
        
    
        transformed_xs = stds_expanded * quantiles_expanded + means_expanded
        
       
        log_det_jacobian = 2 * (jnp.log(2) - transformed_xs - jax.nn.softplus(-2 * transformed_xs))
        
       
        # log_det_sum = jnp.sum(log_det_jacobian, axis=3)
        # print(log_det_jacobian.shape, transformed_xs.shape, means_expanded.shape, stds_expanded.shape, quantiles_expanded.shape)
        
       
        correction = jnp.mean(log_det_jacobian, axis=-1)
        
        
        #When no squashing I need to minimize negative entropy, and when squashing with correction I need to optimize positive entropy
        #How can this be and is there a mistake I am making 
        corrected_entropy = base_entropy + correction
        # corrected_entropy = - correction
        
       
        
       
        summed_entropy = jnp.sum(corrected_entropy, axis=-1) # Shape: (N, T, batch_size)
        
        og_mask = jnp.reshape(mask, (mask.shape[0], mask.shape[1]))  # Remove last dimension if it's 1
        B = summed_entropy.shape[-1]
        batch_idx = jnp.arange(B)  # (B,)

        valid_mask = batch_idx < og_mask[..., None]  # (N, T, B) ← broadcasts correctly
        masked_entropy = jnp.where(valid_mask, summed_entropy, 0.0)  # (N, T, B)
        
        
        
       
        final_entropy = jnp.sum(masked_entropy, axis=-1) / og_mask
        # jax.debug.print("correction shape: {} base {} final {}", correction[0,0], base_entropy[0,0], final_entropy[0,0])
        
        
      
        
        return final_entropy