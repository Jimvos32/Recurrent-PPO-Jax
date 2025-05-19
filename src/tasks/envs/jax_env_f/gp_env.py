import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
# from src.tasks.envs.jax_env_f.jax_function_samplers import EnvParams
from src.tasks.envs.jax_env_f.jax_disp_samplers import EnvParams

from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp

import optax
# import jaxopt

import tqdm

import distrax

def initialize_poly(key: chex.PRNGKey, params: EnvParams, action_dim: int) -> Dict:
    """
    Initializes Polynomial function parameters using config from EnvParams.
    Returns a nested dictionary with dummy arrays for padding.
    """
    # Get Poly specific config from the main EnvParams
    
    poly_config = params.sampler_configs['specific']['poly']
    

    
    # print("poly_config", poly_config)

    key_c, key_weights, key_xmax = jax.random.split(key, 3)
    dim = action_dim # Use concrete integer passed in
    lower, upper = params.x_range # General bounds from params
    degree = poly_config['degree'] # Specific degree
    opt_factor = poly_config['optimum_range_factor'] # Specific factor

    # Sample Poly parameters using bounds from poly_config
    poly_c = jax.random.uniform(key_c, shape=(), minval=poly_config['c_bounds'][0], maxval=poly_config['c_bounds'][1])
    poly_weights = jax.random.uniform(key_weights, shape=(dim,), minval=poly_config['weight_bounds'][0], maxval=poly_config['weight_bounds'][1])
    poly_x_max = jax.random.uniform(key_xmax, shape=(dim,), minval=lower * opt_factor, maxval=upper * opt_factor)
    poly_steep = jax.random.uniform(key_xmax, shape=(), minval=poly_config['steep_min'], maxval=poly_config['steep_max'] * 2.0)

    # Calculate max/min for Poly
    optimum_point = poly_x_max
    max_y = poly_c
    x_min_coords = jnp.where(jnp.abs(poly_x_max - lower) > jnp.abs(upper - poly_x_max), lower, upper)
    diff_min = x_min_coords - poly_x_max
    weighted_term_min = poly_weights * (diff_min ** degree)
    min_y_candidate = poly_c - poly_steep * jnp.sum(weighted_term_min)
    min_y = jnp.minimum(min_y_candidate, max_y - 1e-6)
    
    sam_con = params.sampler_configs
    
    
    sam_con['common'] = {
        'type_index': 1, # We dont know must be overruled
        'optimum_point': optimum_point,
        'max_y': max_y,
        'min_y': min_y,
        'action_dim': dim,
    }
    
    sam_con['specific']['poly']['c'] = poly_c
    sam_con['specific']['poly']['weights'] = poly_weights # Real Array
    sam_con['specific']['poly']['x_max'] = poly_x_max     # Real Array
    sam_con['specific']['poly']['degree'] = degree
    sam_con['specific']['poly']['steepness_factor'] = poly_steep # Real Array
    
   
    # jax.debug.print("init_pol {} {}", params.sampler_configs['common']['type_index'], sam_con['common']['min_y'])
    
    return sam_con


# Add these functions to jax_function_samplers.py

# --- Negative Absolute Value Function ---


# --- compute_y_poly remains the same (accessing nested params) ---
def compute_y_poly(x: chex.Array, sampler_params: Dict, env_params: EnvParams) -> chex.Array:
    # x = jnp.full_like(x, 0.5, dtype=jnp.float32)  # Dummy array for padding
    
    # print("compute_y_poly", x.shape, x.dtype, sampler_params['common']['type_index'])
    poly_params = sampler_params['specific']['poly']
    c = poly_params['c']
    weights = poly_params['weights']
    x_max = poly_params['x_max']
    degree = poly_params['degree']
    steep = poly_params['steepness_factor']
    
    if x.ndim == 1: x = x[jnp.newaxis, :]
    x_max_b = jnp.expand_dims(x_max, axis=0)
    diff = x - x_max_b
   
    weighted_term = weights * (diff ** degree)
    result = c - steep * jnp.sum(weighted_term, axis=-1)
    
    # jax.debug.print("Poly x {} c {} weights {} x_max {} degree {} x_max_b {} diff {} weighted_term {} result {} min{}]",
                    # x, c, weights, x_max, degree, x_max_b, diff, weighted_term, result, sampler_params['common']['min_y'])
    
    # jax.debug.print("\ncompute_y_poly x {} result {} x_max {}\nmin {} max {}\n",
    #                 x, result, x_max, sampler_params['common']['min_y'], sampler_params['common']['max_y'])
    return result.squeeze()



class GP:

    def __init__(
        self, 
        kernel: Callable[[jax.Array, jax.Array], jax.Array], 
        noise: float = 1.0,
        log_noise: bool = False
    ):
        self.kernel = kernel
        self.noise = jax.nn.softplus(noise + 1e-4) if log_noise else noise

    def likelihood_fun(
        self, 
        X: jax.Array, 
        jitter: float = 1e-5
    ) -> distrax.MultivariateNormalFullCovariance:
        gram = self.kernel(X, X)
        return distrax.MultivariateNormalFullCovariance(
            jnp.zeros(len(X)), gram + jnp.eye(len(X)) * (jitter + self.noise)
        )
         
    def posterior_fun(
        self, 
        X: jax.Array, 
        y: jax.Array, 
        jitter: float = 1e-5
    ) -> Callable[[jax.Array], distrax.MultivariateNormalFullCovariance]:        
        
        gram = self.kernel(X, X)
        chol = jnp.linalg.cholesky(gram + jnp.eye(len(y)) * (jitter + self.noise))

        # Representation of Basis functions
        alpha = jnp.linalg.solve(chol.T, jnp.linalg.solve(chol, y))

        def posterior(
            x_test: jax.Array
        ) -> distrax.MultivariateNormalFullCovariance:
            
            k_x = self.kernel(X, x_test)
            k_xx = self.kernel(x_test, x_test)

            v = jnp.linalg.solve(chol, k_x)
            cov = k_xx - v.T @ v

            mean = k_x.T @ alpha

            return distrax.MultivariateNormalFullCovariance(
                mean, cov + jnp.eye(len(mean)) * jitter
            )

        return posterior
         
         
def rbf(x, y, s=5.0, v=0.01):
    return jax.nn.softplus(s) * jnp.exp(-jnp.square(x - y)/(1e-4 + 2 * jax.nn.softplus(v)))

def initialize_gp(key: chex.PRNGKey, params: EnvParams, action_dim: int):

    rbf_kernel = jax.vmap(jax.vmap(rbf, in_axes=(None, 0)), in_axes=(0, None))
    gp = GP(rbf_kernel, 1.0)
    
def compute_gp(gp: GP, xs: chex.Array, ys: chex.Array, resolution: int = 1000):
    post = gp.posterior_fun(xs, ys)
    x_test = jnp.linspace(-7, 10, resolution)
    norm = post(x_test)

    y_test = norm.loc
        