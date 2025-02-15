import flax.linen as nn
import jax.numpy as jnp

from flax.linen.initializers import constant, orthogonal

class Flatten(nn.Module):
    def __call__(self, x):
        return x.reshape(x.shape[0], -1)

def flatten_repr_model():
    def thurn():
        return Flatten()
    return thurn

def atari_conv_repr_model():
    def thurn():
        return nn.Sequential([nn.Conv(32,
                                    kernel_size=(8, 8),
                                    strides=(4, 4),
                                    padding="VALID",
                                    kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0),
                                    ),nn.relu,
                            nn.Conv(
                                    64,
                                    kernel_size=(4, 4),
                                    strides=(2, 2),
                                    padding="VALID",
                                    kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0),
                                ),nn.relu,
                            nn.Conv(
                                    64,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding="VALID",
                                    kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0),
                                ),nn.relu,
                            ])
    return thurn


def mlp_repr_model(hidden_sizes=(256, 128)):
    """
    Creates an MLP-based representation model that extracts features
    from the input to be used by an LSTM.
    
    Args:
        hidden_sizes (tuple): Defines the sizes of the hidden layers.
    
    Returns:
        Function that initializes the MLP model.
    """
    def thurn():
        return nn.Sequential([
            nn.Dense(hidden_sizes[0], 
                     kernel_init=orthogonal(jnp.sqrt(2)), 
                     bias_init=constant(0.0)), 
            nn.relu,
            nn.Dense(hidden_sizes[1], 
                     kernel_init=orthogonal(jnp.sqrt(2)), 
                     bias_init=constant(0.0)), 
            nn.relu
        ])
    
    return thurn