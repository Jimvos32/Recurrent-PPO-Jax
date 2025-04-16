import jax.numpy as jnp

class SamplingImplBase:
    def __init__(self, max_batch, action_dim, sampling_distribution=1):
        self.sample_distribution = sampling_distribution
        self.action_dim = action_dim
        self.batch_size = max_batch
    
    """Shared utilities (only apply_padding_mask). All actual sampling logic is abstract."""
    def apply_padding_mask(self, actions, masks, pad_value=-2.0):
        x,n,z = actions.shape
        batch_idx = jnp.arange(n)
        mask = batch_idx < masks.reshape(x,1)
        full = jnp.broadcast_to(mask.reshape(x,n,1),(x,n,z))
        return jnp.where(full, actions, pad_value)
    
    def sampling_differ(self, act_logits, key, masks):
        raise NotImplementedError

    def gaussian_log_prob(self, actions, logits):
        raise NotImplementedError

    def entropy(self, logits, mask, key=None):
        raise NotImplementedError





            
            