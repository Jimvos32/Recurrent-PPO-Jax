import flax.linen as nn
import jax.numpy as jnp
import jax

from typing import Callable
from src.utils import tree_index
from src.models.rnns.rnn import LSTMMultiLayer
from flax.linen.initializers import constant, orthogonal

def actor_model_discete(dense_dim,action_space):
    def thurn():
        print("dense_dim", dense_dim, "action_space", action_space)
        return nn.Sequential([nn.Dense(dense_dim,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0)),nn.tanh,nn.Dense(action_space,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0))])
    return thurn


def critic_model(dense_dim):
    def thurn():
        return nn.Sequential([nn.Dense(dense_dim,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0)),nn.tanh,nn.Dense(1,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0)),lambda x:jnp.squeeze(x,axis=-1)])
    return thurn


def actor_model_continuous(dense_dim, action_dim):
    """Actor model for continuous action spaces that outputs logits in a compatible format."""
    class ContinuousActor(nn.Module):
        @nn.compact
        def __call__(self, x):
            # Shared features
            print("dense_dim", dense_dim, "action_space", action_dim, "input", x.shape)
            
            x = nn.Dense(dense_dim,
                        kernel_init=orthogonal(jnp.sqrt(2)),
                        bias_init=constant(0.0))(x)
            x = nn.tanh(x)
            print("we got past", x.shape)
            # For compatibility, we'll output [mean, log_std] stacked along the last axis
            # This makes the output shape (batch_size, action_dim * 2) which is similar
            # to the discrete case's (batch_size, num_actions)
            mean = nn.Dense(action_dim,
                          kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(x)
            
            log_std = nn.Dense(action_dim,
                             kernel_init=orthogonal(0.01),
                             bias_init=constant(0.0))(x)
            log_std = jnp.clip(log_std, -20.0, 2.0)
            
            print("meaner than mt ", mean.shape, "log_std", log_std.shape)
            # Stack mean and log_std to maintain shape compatibility
            output = jnp.concatenate([mean, log_std], axis=-1)
            
           
            
            return output
    
    return lambda: ContinuousActor()