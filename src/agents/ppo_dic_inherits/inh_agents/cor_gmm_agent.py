import jax, jax.numpy as jnp
from src.agents.ppo_dic_inherits.inh_agents.standard_sampling import SamplingImplBase

class CorrelatedGaussianMixture(SamplingImplBase):
    
    def sampling_differ(self, act_logits, key, masks):
        N = act_logits.shape[0]

        weight_logits_size = self.sample_distribution * self.action_dim * self.batch_size
        weight_logits = act_logits[..., :weight_logits_size]
        gmm_params = act_logits[..., weight_logits_size:]

        # === Updated Reshaping ===
        #Check if this is the correct way to reshape such that the batches are seperated
        weight_logits = jnp.reshape(weight_logits, (N, self.batch_size, self.action_dim, self.sample_distribution))
        means, std = jnp.split(gmm_params, 2, axis=-1)
        
        # jax.debug.print("weigst {}", weight_logits[0])
        
        
        means_all = jnp.reshape(means, (N, 1, self.sample_distribution))      
        means_all = jnp.expand_dims(jnp.broadcast_to(means_all, (N, self.batch_size, self.sample_distribution)), axis=2)
        means_all = jnp.broadcast_to(means_all, (N, self.batch_size, self.action_dim, self.sample_distribution))
        scale_all = jnp.reshape(std, (N, 1, self.sample_distribution))  # shape: (N, action_dim, k)
        scale_all = jnp.expand_dims(jnp.broadcast_to(scale_all, (N, self.batch_size, self.sample_distribution)), axis=2)
        scale_all = jnp.broadcast_to(scale_all, (N, self.batch_size, self.action_dim, self.sample_distribution))

        # Random keys
        key_cat, key_noise = jax.random.split(key)
        component_indices = jax.random.categorical(key_cat, logits=weight_logits, axis=-1)

        # Indexing arrays
        batch_indices = jnp.arange(N)[:, None, None]
        batch_indices = jnp.broadcast_to(batch_indices, component_indices.shape)

        sample_indices = jnp.arange(self.batch_size)[None, :, None]
        sample_indices = jnp.broadcast_to(sample_indices, component_indices.shape)

        action_indices = jnp.arange(self.action_dim)[None, None, :]
        action_indices = jnp.broadcast_to(action_indices, component_indices.shape)
        

        # Gather chosen means and stds
        chosen_means = means_all[batch_indices, sample_indices, action_indices, component_indices]
        chosen_scales = scale_all[batch_indices, sample_indices, action_indices, component_indices]
        

        noise = jax.random.normal(key_noise, shape=(N, self.batch_size, self.action_dim))
        acts_tick = chosen_means + chosen_scales * noise

        acts_tick = jnp.tanh(acts_tick)
        

        if masks is not None:
            acts_tick = self.apply_padding_mask(acts_tick, masks)

        return acts_tick


    def gaussian_log_prob(self, actions, act_logits):
        """Compute a stable log-probability for a GMM (including the tanh correction)
            For [N,S,b,a].
            where N is the number of parallel environsment, S is the number of time steps,
            b is the batch size, a is the action dimension.
            This is done by having a single gmm for for each step in each environment, so N * S gmm.
            For all values in a step we have weights that sample from their own gmm.
            The gmm is defined by the means and stds of the gmm, which are sampled from a normal distribution.
        ."""
        epsilon = 1e-6
        act_logits = jnp.reshape(act_logits, (act_logits.shape[0],
                                                act_logits.shape[1],
                                                1,
                                                act_logits.shape[-1]))  
        N, S, _, params = act_logits.shape
        
        #
        weight_logits_size = self.sample_distribution * self.action_dim * self.batch_size
        weight_logits = act_logits[..., :weight_logits_size]
        weight_logits = jnp.reshape(weight_logits, (N, S, self.batch_size, self.action_dim, self.sample_distribution))
        
        
        gmm_params = act_logits[..., weight_logits_size:]
                
        # Extract means and std values from gmm_params
        means, std = jnp.split(gmm_params, 2, axis=-1)
        
        means_all = jnp.reshape(means, (N, S, 1, self.sample_distribution))      
        means_all = jnp.expand_dims(jnp.broadcast_to(means_all, (N, S, self.batch_size, self.sample_distribution)), axis=3)
        means_all = jnp.broadcast_to(means_all, (N, S, self.batch_size, self.action_dim, self.sample_distribution))
        scale_all = jnp.reshape(std, (N, S, 1, self.sample_distribution))  # shape: (N, action_dim, k)
        scale_all = jnp.expand_dims(jnp.broadcast_to(scale_all, (N, S, self.batch_size, self.sample_distribution)), axis=3)
        scale_all = jnp.broadcast_to(scale_all, (N, S, self.batch_size, self.action_dim, self.sample_distribution))
        
        # Unsquash the actions: a = tanh(u)  => u = atanh(a)
        u = jnp.arctanh(jnp.clip(actions, -1 + epsilon, 1 - epsilon))
        u_expanded = jnp.expand_dims(u, axis=-1)  # shape: (N, batch_size, action_dim, 1)
        u_expanded = jnp.broadcast_to(u_expanded, (N, S, self.batch_size, self.action_dim, self.sample_distribution))
        
        
        # Compute the log probability for each mixture component.
        
        #This is where I am unsure about, since the shape of means and std are [N, S, batch_size, action_dim, sample_distribution]
        # and the shape of u_expanded is [N, S, batch_size, action_dim, 1]. And the output is [N, S, batch_size, action_dim, sample_distribution]
        #So is the u_expanded expanded in this calculation or should I still do that manually?
        
        log_probs_components = jax.scipy.stats.norm.logpdf(u_expanded, means_all, scale_all)
        # print("u_expanded shape", u_expanded.shape, "means_all shape", means_all.shape, "scale_all shape", scale_all.shape, log_probs_components.shape)
        
        #this part I do not completely get how does symply logging over the weights and adding work
        log_weights = jax.nn.log_softmax(weight_logits, axis=-1)
        
        # print("log_weights prob shape", log_weights.shape, "weight_logits prob shape", weight_logits.shape)
        
        log_component = log_weights + log_probs_components

        # Aggregate the per-component densities using logsumexp over the mixture dimension.
        log_prob_per_dim = jax.scipy.special.logsumexp(log_component, axis=-1)  # shape: (N, batch_size, action_dim)
        # print(log_prob_per_dim.shape)
        base_log_prob = jnp.sum(log_prob_per_dim, axis=-1)      # sum over action dimensions

        # Compute the log-determinant of the tanh Jacobian:
        # A numerically stable expression: 2 * (log(2) - u - softplus(-2*u))
        log_det_jacobian = jnp.sum(2 * (jnp.log(2) - u - jax.nn.softplus(-2 * u)), axis=-1)
        
        log_prob = base_log_prob - log_det_jacobian
        
        valid_mask = (actions[..., 0] != -2)
        log_prob = jnp.where(valid_mask, log_prob, 0.0)
        log_prob = jnp.sum(log_prob, axis=-1)  # shape (N, T)
        
        return log_prob

 
        
    def entropy(self, logits, mask, key=None):
        """
        Compute entropy for a GMM with tanh transformation using a quantile-based
        approach for the Jacobian correction.
        """
        epsilon = 1e-6
        act_logits = jnp.reshape(logits, (logits.shape[0],
                                                logits.shape[1],
                                                1,
                                                logits.shape[-1]))  
        N, S, _, params = act_logits.shape
        
        # Parse logits into weights and GMM parameters
        weight_logits_size = self.sample_distribution * self.action_dim * self.batch_size
        weight_logits = act_logits[..., :weight_logits_size]
        weight_logits = jnp.reshape(weight_logits, (N, S, self.batch_size, self.action_dim, self.sample_distribution))
        
        gmm_params = act_logits[..., weight_logits_size:]
                
        # Extract means and std values from gmm_params
        means, std = jnp.split(gmm_params, 2, axis=-1)
        
        means_all = jnp.reshape(means, (N, S, 1, self.sample_distribution))      
        means_all = jnp.expand_dims(jnp.broadcast_to(means_all, (N, S, self.batch_size, self.sample_distribution)), axis=3)
        means_all = jnp.broadcast_to(means_all, (N, S, self.batch_size, self.action_dim, self.sample_distribution))
        scale_all = jnp.reshape(std, (N, S, 1, self.sample_distribution))
        scale_all = jnp.expand_dims(jnp.broadcast_to(scale_all, (N, S, self.batch_size, self.sample_distribution)), axis=3)
        scale_all = jnp.broadcast_to(scale_all, (N, S, self.batch_size, self.action_dim, self.sample_distribution))
        
        # Calculate mixture weights
        log_weights = jax.nn.log_softmax(weight_logits, axis=-1)
        weights = jnp.exp(log_weights)
        
        # === GMM base entropy calculation ===
        # For each Gaussian component, compute the base entropy
        component_entropy = 0.5 * (jnp.log(2 * jnp.pi * jnp.e) + 2 * jnp.log(scale_all))
        
        # Weighted sum of component entropies + mixture entropy
        mixture_entropy = -jnp.sum(weights * jnp.log(weights + epsilon), axis=-1)
        weighted_component_entropy = jnp.sum(weights * component_entropy, axis=-1)
        base_entropy = mixture_entropy + weighted_component_entropy
        
        # === Jacobian correction using quantile-based approach ===
        n_quantiles = 100
        qs = jnp.linspace(0.01, 0.99, n_quantiles)
        std_normal_quantiles = jax.scipy.stats.norm.ppf(qs)
        
        # Initialize arrays to store corrections
        all_corrections = []
        
        # For each mixture component
        for k in range(self.sample_distribution):
            # Extract this component's means and scales
            component_means = means_all[..., k]  # (N, S, batch_size, action_dim)
            component_scales = scale_all[..., k]  # (N, S, batch_size, action_dim)
            component_weights = weights[..., k]   # (N, S, batch_size, action_dim)
            
            # Expand dimensions for quantile calculation
            means_expanded = jnp.expand_dims(component_means, axis=-1)    # (N, S, batch_size, action_dim, 1)
            scales_expanded = jnp.expand_dims(component_scales, axis=-1)  # (N, S, batch_size, action_dim, 1)
            weights_expanded = jnp.expand_dims(component_weights, axis=-1)  # (N, S, batch_size, action_dim, 1)
            quantiles_expanded = jnp.reshape(std_normal_quantiles, (1, 1, 1, 1, n_quantiles))
            
            # Transform quantiles to this component's distribution
            transformed_xs = scales_expanded * quantiles_expanded + means_expanded
            
            # Calculate log det Jacobian for tanh transformation
            log_det_jacobian = 2 * (jnp.log(2) - transformed_xs - jax.nn.softplus(-2 * transformed_xs))
            
            # Average over quantiles
            component_correction = jnp.mean(log_det_jacobian, axis=-1) * weights_expanded[..., 0]
            all_corrections.append(component_correction)
        
        # Sum corrections across components
        total_correction = jnp.sum(jnp.stack(all_corrections), axis=0)
        
        # Apply correction to base entropy
        corrected_entropy = base_entropy + total_correction
        
        # Apply masking as in your original code
        summed_entropy = jnp.sum(corrected_entropy, axis=-1)  # Shape: (N, T, batch_size)
        
        og_mask = jnp.reshape(mask, (mask.shape[0], mask.shape[1]))  # Remove last dimension if it's 1
        B = summed_entropy.shape[-1]
        batch_idx = jnp.arange(B)  # (B,)

        valid_mask = batch_idx < og_mask[..., None]  # (N, T, B) ← broadcasts correctly
        masked_entropy = jnp.where(valid_mask, summed_entropy, 0.0)  # (N, T, B)
        
        final_entropy = jnp.sum(masked_entropy, axis=-1) / og_mask
        
        return final_entropy