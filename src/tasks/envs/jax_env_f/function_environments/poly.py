import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..jax_disp_samplers import EnvParams

FUNCTION_NAME = "poly"

def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    """Provides the specific config template for the Polynomial function."""
    f_env_config_key = f"{FUNCTION_NAME}_env"
    func_specific_run_config = run_config.get(f_env_config_key, {})
    default_bounds = run_config.get("bounds", (-5.0, 5.0))

    return {
        'c_val': jnp.nan, # For sampled 'c'
        'weights': jnp.full((action_dim,), jnp.nan),
        'x_max_loc': jnp.full((action_dim,), jnp.nan), # For sampled 'x_max'
        'steepness_factor': jnp.nan, # For sampled 'steepness'
        'degree': int(func_specific_run_config.get("degree", 2)), # Non-sampled, from config
        'bounds': tuple(func_specific_run_config.get("bounds", default_bounds)),
        'optimum_range_factor': func_specific_run_config.get('optimum_range_factor', 0.9),
        'c_bounds': tuple(func_specific_run_config.get('c_bounds', (5.0, 20.0))),
        'weight_bounds': tuple(func_specific_run_config.get('weight_bounds', (0.5, 2.0))),
        'steep_min': func_specific_run_config.get('steep_min', 1.0), # Renamed from steep_bounds
        'steep_max': func_specific_run_config.get('steep_max', 2.0), # Renamed from steep_bounds
    }

def initialize_func(key: chex.PRNGKey,
                    env_params_instance: 'EnvParams',
                    action_dim: int,
                    all_possible_names: List[str]) -> Dict:
    """Initializes Polynomial function parameters."""
    poly_base_config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]
    dim = action_dim

    key_c, key_weights, key_xmax_loc, key_steep = jax.random.split(key, 4)

    lower, upper = poly_base_config['bounds']
    degree = poly_base_config['degree'] # Static degree from config
    opt_factor = poly_base_config['optimum_range_factor']

    # Sample Poly parameters
    poly_c_val = jax.random.uniform(key_c, shape=(), minval=poly_base_config['c_bounds'][0], maxval=poly_base_config['c_bounds'][1])
    poly_weights = jax.random.uniform(key_weights, shape=(dim,), minval=poly_base_config['weight_bounds'][0], maxval=poly_base_config['weight_bounds'][1])
    
    # Ensure x_max_loc is sampled within the appropriate range
    # Corrected sampling range for x_max_loc:
    # If opt_factor = 1.0, range is [lower, upper].
    # If opt_factor < 1.0, range is a subset of [lower, upper] centered within it.
    # Example: bounds [-5, 5], opt_factor 0.8. Range width = 10*0.8 = 8.
    # Centered range: [-4, 4].
    # Min = lower + ( (upper-lower) * (1-opt_factor) / 2 )
    # Max = upper - ( (upper-lower) * (1-opt_factor) / 2 )
    range_width = (upper - lower)
    opt_range_width = range_width * opt_factor
    offset_from_bounds = (range_width - opt_range_width) / 2.0
    
    x_max_loc_min = lower + offset_from_bounds
    x_max_loc_max = upper - offset_from_bounds

    poly_x_max_loc = jax.random.uniform(key_xmax_loc, shape=(dim,),
                                        minval=x_max_loc_min,
                                        maxval=x_max_loc_max)

    poly_steep = jax.random.uniform(key_steep, shape=(), minval=poly_base_config['steep_min'], maxval=poly_base_config['steep_max'])

    optimum_point = poly_x_max_loc
    max_y_val = poly_c_val # Max y is at x_max_loc

    jnp_lower = jnp.array(lower)
    jnp_upper = jnp.array(upper)
    x_min_coords = jnp.where(jnp.abs(poly_x_max_loc - jnp_lower) > jnp.abs(jnp_upper - poly_x_max_loc), jnp_lower, jnp_upper)
    
    diff_min = x_min_coords - poly_x_max_loc
    # Ensure weights are positive if degree is even and we subtract, or handle signs carefully
    # The formula is c - steep * sum(weights * diff^degree).
    # If weights are positive, and steep is positive, then diff^degree contributes negatively.
    weighted_term_min = poly_weights * (diff_min ** degree)
    min_y_candidate = poly_c_val - poly_steep * jnp.sum(weighted_term_min)
    min_y_val = jnp.minimum(min_y_candidate, max_y_val - 1e-9)

    output_common_params = {
        'type_index': all_possible_names.index(FUNCTION_NAME),
        'optimum_point': optimum_point,
        'max_y': max_y_val,
        'min_y': min_y_val,
        'action_dim': dim,
        'bounds': tuple((lower, upper)),
    }

    output_specific_params = jax.tree_util.tree_map(lambda x: x, env_params_instance.sampler_configs['specific'])
    
    output_specific_params[FUNCTION_NAME]['c_val'] = poly_c_val
    output_specific_params[FUNCTION_NAME]['weights'] = poly_weights
    output_specific_params[FUNCTION_NAME]['x_max_loc'] = poly_x_max_loc
    output_specific_params[FUNCTION_NAME]['steepness_factor'] = poly_steep
    # 'degree', 'bounds', 'optimum_range_factor', etc. in output_specific_params[FUNCTION_NAME] remain as per template.

    return {'common': output_common_params, 'specific': output_specific_params}

def compute_y_func(x: chex.Array, sampler_params: Dict, env_params_instance: 'EnvParams') -> chex.Array:
    """Computes Polynomial function value: y = c - steep * sum(w_i * (x_i - x_max_i)^degree)."""
    poly_specific_params = sampler_params['specific'][FUNCTION_NAME]
    
    c_val = poly_specific_params['c_val']
    weights = poly_specific_params['weights']
    x_max_loc = poly_specific_params['x_max_loc']
    degree = poly_specific_params['degree'] # This is an int from config
    steep = poly_specific_params['steepness_factor']

    x_eval = x[jnp.newaxis, :] if x.ndim == 1 else x # Ensure (batch, dim)

    diff = x_eval - x_max_loc # (batch, dim) - (dim,) -> (batch, dim)
    
    # Power term: (x_i - x_max_i)^degree
    # Ensure degree is treated as a static value for JAX if it's used in a way that requires it.
    # For jnp.power, it's fine if 'degree' is a Python int.
    power_term = diff ** degree
    
    weighted_term = weights * power_term # (dim,) * (batch, dim) -> (batch, dim)
    sum_weighted_terms = jnp.sum(weighted_term, axis=-1) # Sum over dimensions -> (batch,)
    
    result = c_val - steep * sum_weighted_terms # Shape (batch,)
    
    return result.squeeze() # Squeeze if original x was 1D
