import jax.numpy as jnp

class SamplingImplBaseJax:
    def __init__(self, max_batch, action_dim, sampling_distribution=1):
        self.sample_distribution = sampling_distribution
        self.action_dim = action_dim
        self.batch_size = max_batch
    
    """Shared utilities (only apply_padding_mask). All actual sampling logic is abstract."""
    def apply_padding_mask(self, actions, mask, pad_value=-2.0):
        m_threshold = mask[0] # Extract the scalar index value from M
        num_rows, num_cols = actions.shape

        # Create an array representing column indices [0, 1, 2, ..., z-1]
        col_indices = jnp.arange(num_cols)

  
        condition_to_keep = col_indices < m_threshold

        # Where condition_to_keep is True, take the value from X.
        # Where condition_to_keep is False (i.e., col_index >= m), use replacement_value.
        # JAX broadcasts the (z,) condition and the scalar replacement_value across X.
        masked_actions = jnp.where(condition_to_keep, actions, pad_value)

        # Optional: Ensure replacement_value has the same dtype as X if needed
        # replacement_val_casted = jnp.array(replacement_value, dtype=X.dtype)
        # replaced_X = jnp.where(condition_to_keep, X, replacement_val_casted)

        return masked_actions
    
    def sampling_differ(self, act_logits, key, masks):
        raise NotImplementedError

    def gaussian_log_prob(self, actions, logits):
        raise NotImplementedError

    def entropy(self, logits, mask, key=None):
        raise NotImplementedError





            
            