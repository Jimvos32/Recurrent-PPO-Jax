import jax.numpy as jnp

class SamplingImplBaseJax:
    def __init__(self, max_batch, action_dim, sampling_distribution=1):
        self.sample_distribution = sampling_distribution
        self.action_dim = action_dim
        self.batch_size = max_batch
    
    """Shared utilities (only apply_padding_mask). All actual sampling logic is abstract."""
    def apply_padding_mask(self, actions, mask, pad_value=-2.0):
        m_threshold = mask[0]

        num_rows, num_cols = actions.shape

        row_indices = jnp.arange(num_rows)

       
        condition_to_keep = row_indices < m_threshold # Shape: (num_rows,)

    
        condition_reshaped = condition_to_keep[:, None] # Shape: (num_rows, 1)
      
        masked_actions = jnp.where(condition_reshaped, actions, pad_value)
       
        return masked_actions
    
    def sampling_differ(self, act_logits, key, masks):
        raise NotImplementedError

    def gaussian_log_prob(self, actions, logits):
        raise NotImplementedError

    def entropy(self, logits, mask, key=None):
        raise NotImplementedError





            
            