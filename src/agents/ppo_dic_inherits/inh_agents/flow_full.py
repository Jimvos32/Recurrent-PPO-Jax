import jax, jax.numpy as jnp
from src.agents.ppo_dic_inherits.inh_agents.standard_sampling import SamplingImplBase
from src.agents.ppo_dic_inherits.inh_agents.standard_sampling_jax import SamplingImplBaseJax
from flowjax.distributions import MultivariateNormal, Transformed, Normal
from flowjax.bijections import Tanh, Chain, AbstractBijection

import time
EPSILON = 1e-6


class FlowGaussian(SamplingImplBaseJax):
    def __init__(self, max_batch, action_dim, sampling_distribution=1):
        """
        Jax implementation for sampling using flowjax with diagonal Normal distributions
        transformed by an optional external flow and then by Tanh.
        The parameters of the `flow_object` (if provided) are managed externally.

        Args:
            max_batch: Batch size (number of parallel distribution instances).
            action_dim: Dimensionality of the action space.
        """
        super().__init__(max_batch, action_dim, sampling_distribution)

  

    def _get_final_bijection_chain(self, flow_object: AbstractBijection | None):
        """
        Constructs the final bijection chain: [Optional Flow, Tanh].
        Args:
            flow_object: The optional flowjax flow instance (e.g., BNAF), or None.
        Returns:
            A flowjax.bijections.AbstractBijection object.
        """
        # tanh_bij = Tanh(shape=(self.action_dim,)) # Tanh squashes to [-1, 1] for each action dimension
        # if flow_object is not None:
        #     # Apply flow_object then Tanh
        #     return Chain([flow_object, tanh_bij])
        
        tanh_bijection = Tanh(shape=(self.action_dim,)) 
        

        if flow_object is not None:
            # Chain the bijections: Input -> flow_object -> Tanh
            # Order matters for Chain: applies first element, then second, etc
            final_bijection = Chain([flow_object, tanh_bijection])
        else:
            # No additional flow module, just Tanh
            #in current setting this is always the case
            final_bijection = tanh_bijection
            
        return final_bijection
        
        
        # return tanh_bij

    def _get_transformed_distribution(self, means, scales, flow_object: AbstractBijection | None):
        """
        Creates the final transformed diagonal Normal distribution.
        The transformation is Normal -> flow_object (optional) -> Tanh.
        Args:
            means: Mean vector for the Normal distribution. Shape (self.action_dim,).
            scales: Scale vector (std devs) for the Normal distribution. Shape (self.action_dim,).
            flow_object: The optional flowjax flow instance, or None.
        Returns:
            A flowjax.distributions.Transformed distribution object.
        """
        # jax.debug.print("means {}, scales {}", means, scales)
        
        diag_normal_dist = Normal(loc=means, scale=scales)
        final_bijection = self._get_final_bijection_chain(flow_object)
        return Transformed(diag_normal_dist, final_bijection)

    def sampling_differ(self, act_logits, flow_object: AbstractBijection | None, key, masks):
      
     
        #Act logist of shape (2 * batch * dim,)
        #Mask threshold array of shape (1,)
        # print("sampling_differ", act_logits.shape, flow_object, key.shape, masks.shape)
        # Example output for act_dim =2, batch size = 3
        # sampling_differ (12,) None () (1,)
        
        
        
        temp_means, temp_scales = jnp.split(act_logits, 2, axis=-1)
        
        # Reshape to (self.batch_size, self.action_dim) using class attributes
        means_batch = temp_means.reshape((self.batch_size, self.action_dim))
        scales_batch = temp_scales.reshape((self.batch_size, self.action_dim))
        
        # scales_batch = jnp.full_like(scales_batch, 1e-6) # Ensure scales are positive
        
        

        # The rest of the function uses these explicitly shaped means_batch and scales_batch
        def sample_from_single_params(m, s, k_):
            # flow_object is captured from the outer scope
            
            dist = self._get_transformed_distribution(m, s, flow_object)
            return dist.sample(k_) 

        keys_for_batch = jax.random.split(key, means_batch.shape[0]) # means_batch.shape[0] is self.batch_size
        samples = jax.vmap(sample_from_single_params)(means_batch, scales_batch, keys_for_batch)
        
        # jax.debug.print("means {} \n samples {}", temp_means, samples)
        
        
        masked_samples = self.apply_padding_mask(samples, masks)
        
        
        # Example output for act_dim =2, batch size = 3
        #sample out (3, 2)
        
        
        
        
        return masked_samples

    def gaussian_log_prob(self, actions, act_logits, flow_object: AbstractBijection | None):
       
        # Example: actions (64, 3, 2), act_logits (64, 12) for N=64, batch_size=3, action_dim=2
        
        print("gaussian_log_prob", actions.shape, act_logits.shape)
        

        clipped_actions = jnp.clip(actions, -1 + EPSILON, 1 - EPSILON)
        
        
        N = act_logits.shape[0] # Rollout length
        
        # Parse act_logits for the entire rollout
        # means_part_flat, raw_scales_part_flat shape: (N, self.batch_size * self.action_dim)
        means_part_flat, raw_scales_part_flat = jnp.split(act_logits, 2, axis=-1)
        
        # Reshape to (N, self.batch_size, self.action_dim)
        means_all = means_part_flat.reshape((N, self.batch_size, self.action_dim))
        scales_all = raw_scales_part_flat.reshape((N, self.batch_size, self.action_dim))
        
        # scales_all = jax.nn.softplus(raw_scales_all) + EPSILON

        # Define log_prob calculation for one component action (action_dim,)
        def get_log_prob_for_one_component(mean_vec, scale_vec, action_vec, flow_obj_inner):
            # mean_vec, scale_vec, action_vec are (self.action_dim,)
            dist = self._get_transformed_distribution(mean_vec, scale_vec, flow_obj_inner)
            return dist.log_prob(action_vec) # scalar log_prob for this component

        # Vmap over N (rollout) and self.batch_size (components of joint action)
        # Input arrays to vmap: means_all, scales_all, clipped_actions
        # These have leading dimensions (N, self.batch_size)
        # flow_object is static (None for in_axes)
        
        
        log_probs_individual_components = jax.vmap(
            jax.vmap(get_log_prob_for_one_component, in_axes=(0, 0, 0, None)), # Inner vmap over self.batch_size
            in_axes=(0, 0, 0, None)                                          # Outer vmap over N
        )(means_all, scales_all, clipped_actions, flow_object)
        # log_probs_individual_components shape: (N, self.batch_size)

        # Masking: if an action component was padded (-2.0), its log_prob is 0.
        # The padding is typically on the first element of the action_dim vector.
        valid_mask_components = (actions[..., 0] != -2.0) # Shape: (N, self.batch_size)
        
        masked_log_probs_components = jnp.where(valid_mask_components, log_probs_individual_components, 0.0)
        
        # Sum log_probs of components to get log_prob of the joint action for each rollout step
        final_log_probs_joint_action = jnp.sum(masked_log_probs_components, axis=-1) # Shape: (N,)
        
        # Expected output shape: (N,), e.g., (64,)
        return final_log_probs_joint_action
    
    def entropy(self, act_logits, mask, flow_object: AbstractBijection | None, key, num_samples_mc=100):
        """
        Estimates the entropy of joint actions using Monte Carlo.
        act_logits: Shape (N, 2 * self.batch_size * self.action_dim).
        mask: Validity mask for rollout steps. Shape (N, 1). 1 for valid, 0 for invalid.
        key: JAX PRNGKey.
        Returns:
            Entropy for each joint action. Shape (N, 1).
        """
        # Example: act_logits (64, 12), mask (64, 1) for N=64, batch_size=3, action_dim=2

        N = act_logits.shape[0] # Rollout length
        
        # Parse act_logits
        means_part_flat, raw_scales_part_flat = jnp.split(act_logits, 2, axis=-1)
        means_all = means_part_flat.reshape((N, self.batch_size, self.action_dim))
        scales_all = raw_scales_part_flat.reshape((N, self.batch_size, self.action_dim))

        # Define MC entropy estimation for one component distribution (action_dim,)
        def estimate_entropy_single_component(mean_vec, scale_vec, flow_obj_inner, subkey_inner):
            dist = self._get_transformed_distribution(mean_vec, scale_vec, flow_obj_inner)
            
            # Sample from the transformed distribution
            samples = dist.sample(subkey_inner, sample_shape=(num_samples_mc,)) # (num_samples_mc, self.action_dim)
            
            # Clip samples before log_prob, consistent with gaussian_log_prob
            clipped_samples = jnp.clip(samples, -1 + EPSILON, 1 - EPSILON)
            
            log_probs_samples = dist.log_prob(clipped_samples) # (num_samples_mc,)
            return -jnp.mean(log_probs_samples) # Scalar entropy estimate

        # Generate keys for each component distribution across rollout and batch
        total_components = N * self.batch_size
        component_keys = jax.random.split(key, total_components).reshape(N, self.batch_size) # each key is typically (2,)
        # component_keys = jax.random.split(key, total_components).reshape(N, self.batch_size, 2)

        # Vmap over N (rollout) and self.batch_size (components)
        entropies_individual_components = jax.vmap(
            jax.vmap(estimate_entropy_single_component, in_axes=(0, 0, None, 0)), # Inner vmap over self.batch_size
            in_axes=(0, 0, None, 0)                                             # Outer vmap over N
        )(means_all, scales_all, flow_object, component_keys)
        # entropies_individual_components shape: (N, self.batch_size)

        # Sum entropies of components to get entropy of the joint action
        # (Entropy of independent variables is the sum of their entropies)
        entropy_joint_action = jnp.sum(entropies_individual_components, axis=-1) # Shape: (N,)
        
        # Apply rollout step mask
        # mask is (N, 1), 1.0 for valid, 0.0 for invalid
        valid_rollout_step_mask = (mask.squeeze(axis=-1) > 0.5) # Shape: (N,)
        
        final_entropy = jnp.where(valid_rollout_step_mask, entropy_joint_action, 0.0) # Shape: (N,)
        
        final_entropy_reshaped = final_entropy[:, None] # Reshape to (N, 1)
        
        # Expected output shape: (N, 1), e.g., (64, 1)
        return final_entropy_reshaped

    # def get_pdf(self, act_logits_one_dist, flow_object: AbstractBijection | None, x_values):
        
    #     print("act_logits_one_dist", act_logits_one_dist.shape, flow_object, x_values.shape)
    #     # act_logits_one_dist: Shape (2 * self.self.batch) since this is only called with self.action_dim=1
    #     # Example: act_logits_one_dist (6,), flow_object None, x_values (200, 1)

      
    #     # Now params_for_dist has shape (2 * self.action_dim,)
    #     mean_vec_flat, raw_scale_vec_flat = jnp.split(act_logits_one_dist, 2, axis=-1)
        
    #     # Reshape to (self.action_dim,).
    #     mean_vec = mean_vec_flat.reshape((self.batch_size, self.action_dim,)) 
    #     scale_vec = raw_scale_vec_flat.reshape((self.batch_size, self.action_dim,))
        
        
    #     # _get_transformed_distribution uses self.action_dim internally for Tanh shape and Normal params
    #     dist = self._get_transformed_distribution(mean_vec, scale_vec, flow_object)
        
      
    #     # x_values are in the target space (after Tanh, so in [-1, 1])
    #     log_pdf_values = dist.log_prob(x_values) # Shape (num_points,)
    #     pdf_values = jnp.exp(log_pdf_values)
        
    #     print("how would this be with ")
    #     # Expected output shape: (num_points,), e.g., (200,)
    #     return pdf_values
    
    def get_pdf(self, act_logits_one_dist, flow_object: AbstractBijection | None, x_values):
      
        # Example: self.batch_size = 3, self.action_dim = 1
        # act_logits_one_dist.shape should be (6,)
        # x_values.shape could be (200, 1)
        # Expected output shape: (3, 200)

        if self.action_dim != 1:
            # This warning is important because _get_transformed_distribution and _get_final_bijection_chain
            # use self.action_dim. If it's not 1, the interpretation of "1D distribution" is violated.
            print(f"Critical Warning: FlowGaussian.get_pdf assumes self.action_dim=1 for creating 1D distributions. Current self.action_dim={self.action_dim}. Results may be incorrect.")

        # Split flattened logits into means and scales for the batch
        # Each will have shape (self.batch_size,)
        mean_params_flat, raw_scale_params_flat = jnp.split(act_logits_one_dist, 2, axis=-1)

        # Reshape to (self.batch_size, self.action_dim) to clearly separate params for each distribution.
        # Since self.action_dim is assumed to be 1 here, this becomes (self.batch_size, 1).
        means_for_batch = mean_params_flat.reshape((self.batch_size, self.action_dim))
        scales_for_batch = raw_scale_params_flat.reshape((self.batch_size, self.action_dim))
        
        # Ensure scales are positive. Apply softplus or similar if raw_scale_params_flat are direct NN outputs.
        # For example: scales_for_batch = jax.nn.softplus(scales_for_batch) + EPSILON
        # Assuming for now they are already valid positive scales.

        # Define a function to get PDF for a single distribution's parameters
        # over all x_values.
        def get_pdf_single_dist(mean_vec, scale_vec, x_eval_points, flow_obj_inner):
            # mean_vec, scale_vec are for one distribution, shape (self.action_dim,) which is (1,)
            # x_eval_points has shape (num_x_points, self.action_dim) which is (200,1)
            
            # Create the single 1D transformed distribution
            dist = self._get_transformed_distribution(mean_vec, scale_vec, flow_obj_inner)
            
            # log_prob will operate on each point in x_eval_points
            # Input to log_prob: x_eval_points of shape (num_x_points, 1)
            # Output of log_prob: shape (num_x_points,)
            
            log_pdf_values = dist.log_prob(x_eval_points)
            return jnp.exp(log_pdf_values)

        # Vectorize the calculation over the batch of means and scales.
        # x_values and flow_object are the same for all distributions in the batch.
        # means_for_batch has shape (self.batch_size, 1)
        # scales_for_batch has shape (self.batch_size, 1)
        # We vmap over the first axis (the batch dimension) of means and scales.
        # `in_axes=(0, 0, None, None)` means:
        #   - Take the i-th element from means_for_batch
        #   - Take the i-th element from scales_for_batch
        #   - Pass x_values as is (None means don't map over this argument)
        #   - Pass flow_object as is
        batched_pdfs = jax.vmap(get_pdf_single_dist, in_axes=(0, 0, None, None))(
            means_for_batch, scales_for_batch, x_values, flow_object
        )
        # Output shape: (self.batch_size, num_x_points)

        return batched_pdfs