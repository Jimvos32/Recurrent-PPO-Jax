from src.model_fns.achead_fns import MLP, PolicyParameterClipping
import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Callable
import math

class PredictorModel(nn.Module):
    recon_hidden_layer: list[int]  # List of hidden layer sizes for the MLP
    out:int

  
    @nn.compact
    def __call__(self, hidden, actions):
        
        # print("hidden:", hidden.shape, "actions:", actions.shape)
        
        inp = jnp.concatenate([hidden, actions], axis=1)
        
        # print("hidd:", hidden.shape, "actopm", actions.shape, "comb", inp.shape)
        
        
        reconstructor = MLP(hidden_sizes=self.recon_hidden_layer, output_size=self.out)
        target = reconstructor(inp)
        
        
        
        return target
    
    
    
class PredictorModelProb(nn.Module):
    recon_hidden_layer: list[int]  # List of hidden layer sizes for the MLP
    out:int

  
    @nn.compact
    def __call__(self, hidden, actions):
        
        # print("hidden:", hidden.shape, "actions:", actions.shape)
        
        inp = jnp.concatenate([hidden, actions], axis=1)
        
        # print("hidd:", hidden.shape, "actopm", actions.shape, "comb", inp.shape)
        
        
        reconstructor_mu = MLP(hidden_sizes=self.recon_hidden_layer, output_size=self.out)
        reconstructor_std = MLP(hidden_sizes=self.recon_hidden_layer, output_size=self.out)
        target_mu = reconstructor_mu(inp)
        target_std = reconstructor_std(inp)
        
        projection = PolicyParameterClipping()
        means, stds = projection(target_mu, target_std)
        
        output = jnp.concatenate([means, stds], axis=-1)
        
        
        
        return output
    
    
    

# class MLP(nn.Module):
#     hidden_sizes: tuple
#     output_size: int
#     activation: Callable = nn.relu

#     @nn.compact
#     def __call__(self, x):
#         # Process hidden layers with the provided activation function.
#         for size in self.hidden_sizes:
#             x = nn.Dense(
#                 size,
#                 kernel_init=orthogonal(jnp.sqrt(2)),
#                 bias_init=constant(0.0)
#             )(x)
#             x = self.activation(x)
#         # Final output layer (usually without an activation function)
#         x = nn.Dense(
#             self.output_size,
#             kernel_init=orthogonal(0.8),
#             bias_init=constant(0.0)
#         )(x)
#         return x