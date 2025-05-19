import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any

# Assuming EnvParams is defined in a central place and imported if needed by these functions
# For example, if they need env_params_instance.x_range or other global settings.
# from ..jax_disp_samplers import EnvParams # Or wherever EnvParams is defined
# However, the current initialize_XYZ functions take EnvParams directly.

# If ALL_POSSIBLE_FUNCTION_NAMES is needed for type_index, it should be imported or passed.
# For simplicity, let's assume a global registry will handle the string-to-index mapping.
# For now, we can hardcode the name or expect it to be set by the central registry.

FUNCTION_NAME = "ackley"

def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    """Provides the specific config template for Ackley."""
    f_env_config_key = f"{FUNCTION_NAME}_env"
    func_specific_run_config = run_config.get(f_env_config_key, {})
    return {
        "a": jnp.nan, "b": jnp.nan, "c": jnp.nan,
        "bounds": tuple(func_specific_run_config.get("bounds", run_config.get("bounds", (-5.0, 5.0)))),
        "optimum_range_factor": func_specific_run_config.get('optimum_range_factor', 0.9),
        "a_bounds": tuple(func_specific_run_config.get('a_bounds', (15.0, 20.0))),
        "b_bounds": tuple(func_specific_run_config.get('b_bounds', (0.1, 0.2))),
        "c_bounds": tuple(func_specific_run_config.get('c_bounds', (2 * jnp.pi, 2 * jnp.pi))),
    }

def initialize_func(key: chex.PRNGKey, env_params_instance: 'EnvParams', action_dim: int, all_possible_names: list) -> Dict:
    """Initializes Ackley function."""
    # env_params_instance is the full EnvParams dataclass instance
    ackley_base_config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]

    key_a, key_b, key_c, key_opt = jax.random.split(key, 4)
    dim = action_dim

    lower_bound_scalar, upper_bound_scalar = ackley_base_config['bounds']
    opt_factor = ackley_base_config['optimum_range_factor']
    center_range = (upper_bound_scalar - lower_bound_scalar) * opt_factor
    offset = (upper_bound_scalar - lower_bound_scalar - center_range) / 2.0

    a = jax.random.uniform(key_a, shape=(), minval=ackley_base_config['a_bounds'][0], maxval=ackley_base_config['a_bounds'][1])
    b = jax.random.uniform(key_b, shape=(), minval=ackley_base_config['b_bounds'][0], maxval=ackley_base_config['b_bounds'][1])
    c = jax.random.uniform(key_c, shape=(), minval=ackley_base_config['c_bounds'][0], maxval=ackley_base_config['c_bounds'][1])
    optimum_point = jax.random.uniform(key_opt, shape=(dim,),
                                      minval=lower_bound_scalar + offset,
                                      maxval=upper_bound_scalar - offset,
                                      dtype=jnp.float64) # Ensure dtype

    max_y_val = 0.0
    mid_point_of_bounds = (lower_bound_scalar + upper_bound_scalar) / 2.0
    furthest_corner_coords = jnp.where(optimum_point > mid_point_of_bounds,
                                       lower_bound_scalar,
                                       upper_bound_scalar)
    z = furthest_corner_coords - optimum_point
    sum_sq_term_val = jnp.sum(z**2) / jnp.maximum(dim, 1) # Avoid div by zero if dim could be 0
    cos_sum_term_val = jnp.sum(jnp.cos(c * z)) / jnp.maximum(dim, 1)
    ackley_val_at_furthest_corner = -a * jnp.exp(-b * jnp.sqrt(sum_sq_term_val)) - jnp.exp(cos_sum_term_val) + a + jnp.exp(1.0)
    min_y_val = jnp.minimum(-ackley_val_at_furthest_corner, max_y_val - 1e-9)

    output_common_params = {
        'type_index': all_possible_names.index(FUNCTION_NAME), # Global 0-based index
        'optimum_point': optimum_point,
        'max_y': max_y_val,
        'min_y': min_y_val,
        'action_dim': dim,
        'bounds': tuple((float(lower_bound_scalar), float(upper_bound_scalar))),
    }

    output_specific_params = jax.tree_util.tree_map(lambda x: x, env_params_instance.sampler_configs['specific'])
    # Update only Ackley's specific sampled parameters
    output_specific_params[FUNCTION_NAME]['a'] = a
    output_specific_params[FUNCTION_NAME]['b'] = b
    output_specific_params[FUNCTION_NAME]['c'] = c
    # Other params in output_specific_params[FUNCTION_NAME] (like bounds, optimum_range_factor) remain as configured in template.

    return {'common': output_common_params, 'specific': output_specific_params}


def compute_y_func(x: chex.Array, sampler_params: Dict, env_params_instance: 'EnvParams') -> chex.Array:
    """Computes Ackley function value."""
    common_params = sampler_params['common']
    # Specific params for ackley are in sampler_params['specific']['ackley']
    ackley_specific_params = sampler_params['specific'][FUNCTION_NAME]

    optimum_point = common_params['optimum_point']
    dim = common_params['action_dim']
    a = ackley_specific_params['a']
    b = ackley_specific_params['b']
    c = ackley_specific_params['c']

    # Ensure x is at least 2D (batch, dim) for consistent axis operations
    if x.ndim == 1: x_b = x[jnp.newaxis, :]
    else: x_b = x
    
    optimum_point_b = jnp.expand_dims(optimum_point, axis=0)
    z = x_b - optimum_point_b
    
    sum_sq_term = -a * jnp.exp(-b * jnp.sqrt(jnp.sum(z**2, axis=-1) / jnp.maximum(dim,1)))
    cos_term = -jnp.exp(jnp.sum(jnp.cos(c * z), axis=-1) / jnp.maximum(dim,1))
    y = sum_sq_term + cos_term + a + jnp.exp(1.0)
    
    result = -y # Negate since we want to maximize (original Ackley is minimized at 0)
    return result.squeeze() # Squeeze in case input x was 1D and became (1,1) -> (1,) -> scalar