import flax.linen as nn
import jax.numpy as jnp
import jax

from typing import Callable
from src.utils import tree_index
from src.models.rnns.rnn import LSTMMultiLayer
from flax.linen.initializers import constant, orthogonal

def actor_model_discete(dense_dim,action_space):
    def thurn():
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
            # print("in", x.shape)
            
            x = nn.Dense(dense_dim,
                        kernel_init=orthogonal(jnp.sqrt(2)),
                        bias_init=constant(0.0))(x)
            # print("out", x.shape)
            x = nn.tanh(x)
            # For compatibility, we'll output [mean, log_std] stacked along the last axis
            # This makes the output shape (batch_size, action_dim * 2) which is similar
            # to the discrete case's (batch_size, num_actions)
            mean = nn.Dense(action_dim[0],
                          kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(x)
            
            log_std = nn.Dense(action_dim[0],
                             kernel_init=orthogonal(0.01),
                             bias_init=constant(0.0))(x)
            
            log_std = jnp.clip(log_std, -20.0, 2.0)
            
            
            # print("policy mean out ", mean.shape, "policy std out ", log_std.shape)
            # Stack mean and log_std to maintain shape compatibility
            output = jnp.concatenate([mean, log_std], axis=-1)
            # print("single_pol_output", output.shape)
            
            return output
    return lambda: ContinuousActor()
        
        
def actor_model_gmm(dense_dim, sample_distribution):
    """Actor model for continuous action spaces that outputs logits in a compatible format."""
    class ContinuousActor(nn.Module):
        @nn.compact
        def __call__(self, x):
            # Shared features
            # print("in", x.shape)
            
            x = nn.Dense(dense_dim,
                        kernel_init=orthogonal(jnp.sqrt(2)),
                        bias_init=constant(0.0))(x)
            # print("out", x.shape)
            x = nn.tanh(x)
            # For compatibility, we'll output [mean, log_std] stacked along the last axis
            # This makes the output shape (batch_size, action_dim * 2) which is similar
            # to the discrete case's (batch_size, num_actions)
            mean = nn.Dense(sample_distribution,
                          kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(x)
            
            log_std = nn.Dense(sample_distribution,
                             kernel_init=orthogonal(0.01),
                             bias_init=constant(0.0))(x)
            
            log_std = jnp.clip(log_std, -20.0, 2.0)
            
            weights = nn.Dense(sample_distribution,
                             kernel_init=orthogonal(0.01),
                             bias_init=constant(0.0))(x)
            
            
            
            # print("policy mean out ", mean.shape, "policy std out ", log_std.shape)
            # Stack mean and log_std to maintain shape compatibility
            output = jnp.concatenate([mean, log_std, weights], axis=-1)
            # print("single_pol_output", output.shape)
            
            return output
    
    return lambda: ContinuousActor()


def actor_model_gmm(dense_dim, sample_distribution):
    """Actor model for continuous action spaces that outputs logits in a compatible format."""
    class ContinuousActor(nn.Module):
        @nn.compact
        def __call__(self, x):
            # Shared features
            # print("in", x.shape)
            
            x = nn.Dense(dense_dim,
                        kernel_init=orthogonal(jnp.sqrt(2)),
                        bias_init=constant(0.0))(x)
            # print("out", x.shape)
            x = nn.tanh(x)
            # For compatibility, we'll output [mean, log_std] stacked along the last axis
            # This makes the output shape (batch_size, action_dim * 2) which is similar
            # to the discrete case's (batch_size, num_actions)
            mean = nn.Dense(sample_distribution,
                          kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(x)
            
            log_std = nn.Dense(sample_distribution,
                             kernel_init=orthogonal(0.01),
                             bias_init=constant(0.0))(x)
            
            log_std = jnp.clip(log_std, -20.0, 2.0)
            
            weights = nn.Dense(sample_distribution,
                             kernel_init=orthogonal(0.01),
                             bias_init=constant(0.0))(x)
            
            
            
            # print("policy mean out ", mean.shape, "policy std out ", log_std.shape)
            # Stack mean and log_std to maintain shape compatibility
            output = jnp.concatenate([mean, log_std, weights], axis=-1)
            # print("single_pol_output", output.shape)
            
            return output
    
    return lambda: ContinuousActor()


def actor_model_continuous_params(shared_hidden_sizes=(256, 128), policy_hidden_sizes=(64, 32, 2)):
    """Actor model for continuous action spaces with parameterizable layer sizes.
    
    Args:
        shared_hidden_sizes: Tuple of hidden layer sizes for the shared network
        policy_hidden_sizes: Tuple of hidden layer sizes for both mean and log_std networks,
                            with the last value being used as the action dimension
    
    Returns:
        A function that returns a ContinuousActor module
    """
    # Extract action dimension from the last element of policy_hidden_sizes
    action_dim = policy_hidden_sizes[-1]
    
    class ContinuousActor(nn.Module):
        @nn.compact
        def __call__(self, x):
            # Helper function to create an MLP with given hidden sizes
            def create_mlp(inputs, hidden_sizes, final_activation=None, kernel_init_scale=jnp.sqrt(2), bias_init_val=0.0):
                x = inputs
                # Process all hidden layers except the last one
                for size in hidden_sizes:
                    x = nn.Dense(size,
                                kernel_init=orthogonal(kernel_init_scale),
                                bias_init=constant(bias_init_val))(x)
                    x = nn.tanh(x)
                return x
            
            # Shared feature network
            x = create_mlp(x, shared_hidden_sizes)
            
            # Create separate networks for mean and log_std, both with the same architecture
            # Use all but the last element of policy_hidden_sizes for the hidden layers
            policy_layers = policy_hidden_sizes[:-1]
            
            # Mean network
            mean_hidden = create_mlp(x, policy_layers)
            mean = nn.Dense(action_dim,
                          kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(mean_hidden)
            
            # Log std network - same architecture as mean network
            log_std_hidden = create_mlp(x, policy_layers)
            log_std = nn.Dense(action_dim,
                             kernel_init=orthogonal(0.01),
                             bias_init=constant(0.0))(log_std_hidden)
            
            # Clip log standard deviation for numerical stability
            log_std = jnp.clip(log_std, -20.0, 2.0)
            
            # Stack mean and log_std to maintain shape compatibility
            output = jnp.concatenate([mean, log_std], axis=-1)
            
            return output
    
    return lambda: ContinuousActor()


def actor_gmm_params(shared_hidden_sizes=(256, 128), policy_hidden_sizes=(64, 32, 2)):
    """Actor model for continuous action spaces with parameterizable layer sizes.
    
    Args:
        shared_hidden_sizes: Tuple of hidden layer sizes for the shared network
        policy_hidden_sizes: Tuple of hidden layer sizes for both mean and log_std networks,
                            with the last value being used as the action dimension
    
    Returns:
        A function that returns a ContinuousActor module
    """
    # Extract action dimension from the last element of policy_hidden_sizes
    action_dim = policy_hidden_sizes[-1]
    
    class ContinuousActor(nn.Module):
        @nn.compact
        def __call__(self, x):
            # Helper function to create an MLP with given hidden sizes
            def create_mlp(inputs, hidden_sizes, final_activation=None, kernel_init_scale=jnp.sqrt(2), bias_init_val=0.0):
                x = inputs
                # Process all hidden layers except the last one
                for size in hidden_sizes:
                    x = nn.Dense(size,
                                kernel_init=orthogonal(kernel_init_scale),
                                bias_init=constant(bias_init_val))(x)
                    x = nn.tanh(x)
                return x
            
            # Shared feature network
            x = create_mlp(x, shared_hidden_sizes)
            
            # Create separate networks for mean and log_std, both with the same architecture
            # Use all but the last element of policy_hidden_sizes for the hidden layers
            policy_layers = policy_hidden_sizes[:-1]
            
            # Mean network
            mean_hidden = create_mlp(x, policy_layers)
            mean = nn.Dense(action_dim,
                          kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(mean_hidden)
            
            # Log std network - same architecture as mean network
            log_std_hidden = create_mlp(x, policy_layers)
            log_std = nn.Dense(action_dim,
                             kernel_init=orthogonal(0.01),
                             bias_init=constant(0.0))(log_std_hidden)
            
            # Clip log standard deviation for numerical stability
            log_std = jnp.clip(log_std, -20.0, 2.0)
            
            weights = nn.Dense(action_dim,
                             kernel_init=orthogonal(0.01),
                             bias_init=constant(0.0))(x)
            
            
            
            # print("policy mean out ", mean.shape, "policy std out ", log_std.shape)
            # Stack mean and log_std to maintain shape compatibility
            output = jnp.concatenate([mean, log_std, weights], axis=-1)
            # print("single_pol_output", output.shape)
            
            return output
    
    return lambda: ContinuousActor()