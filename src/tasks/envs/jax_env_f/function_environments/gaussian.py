import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any, List, TYPE_CHECKING

# For type checking EnvParams to avoid circular import at runtime
if TYPE_CHECKING:
    from ..jax_disp_samplers import EnvParams # Adjust path as per your final structure

FUNCTION_NAME = "gaussian"

def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    """Provides the specific config template for the Gaussian function."""
    f_env_config_key = f"{FUNCTION_NAME}_env"
    func_specific_run_config = run_config.get(f_env_config_key, {})

    # Default bounds for this function, can be overridden by run_config
    default_bounds = run_config.get("bounds", (-5.0, 5.0)) # Global default

    return {
        'center': jnp.full((action_dim,), jnp.nan),
        'width': jnp.full((action_dim,), jnp.nan),
        'amplitude': jnp.nan,
        'baseline': jnp.nan,
        'bounds': tuple(func_specific_run_config.get("bounds", default_bounds)),
        'center_opt_factor': func_specific_run_config.get('center_opt_factor', 0.9),
        'width_bounds': tuple(func_specific_run_config.get('width_bounds', (0.1, 2.0))),
        'amplitude_bounds': tuple(func_specific_run_config.get('amplitude_bounds', (0.1, 2.0))),
        'baseline_bounds': tuple(func_specific_run_config.get('baseline_bounds', (-10.0, 10.0))),
    }

def initialize_func(key: chex.PRNGKey,
                    env_params_instance: 'EnvParams',
                    action_dim: int,
                    all_possible_names: List[str]) -> Dict:
    """Initializes Gaussian function parameters."""
    gaussian_base_config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]
    dim = action_dim # Static argument

    key_center, key_width, key_amp, key_base = jax.random.split(key, 4)

    lower, upper = gaussian_base_config['bounds'] # Bounds specific to Gaussian config
    opt_factor = gaussian_base_config['center_opt_factor']

    center = jax.random.uniform(key_center, shape=(dim,),
                                minval=lower * opt_factor,
                                maxval=upper * opt_factor, dtype=jnp.float64)
    width = jax.random.uniform(key_width, shape=(dim,),
                               minval=gaussian_base_config['width_bounds'][0],
                               maxval=gaussian_base_config['width_bounds'][1], dtype=jnp.float64)
    amplitude = jax.random.uniform(key_amp, shape=(),
                                  minval=gaussian_base_config['amplitude_bounds'][0],
                                  maxval=gaussian_base_config['amplitude_bounds'][1], dtype=jnp.float64)
    baseline = jax.random.uniform(key_base, shape=(),
                                  minval=gaussian_base_config['baseline_bounds'][0],
                                  maxval=gaussian_base_config['baseline_bounds'][1], dtype=jnp.float64)

    optimum_point = center
    max_y_val = baseline + amplitude # Max y occurs at the center

    # Min y occurs at the boundary point furthest from the center
    # Ensure lower/upper are JAX arrays for jnp.where
    jnp_lower = jnp.array(lower, dtype=jnp.float64)
    jnp_upper = jnp.array(upper, dtype=jnp.float64)
    x_min_coords = jnp.where(jnp.abs(center - jnp_lower) > jnp.abs(jnp_upper - center), jnp_lower, jnp_upper)
    
    diff_min_sq = ((x_min_coords - center) / jnp.maximum(width, 1e-9)) ** 2 # Avoid division by zero in width
    exponent_at_min = -0.5 * jnp.sum(diff_min_sq)
    min_y_candidate = baseline + amplitude * jnp.exp(exponent_at_min)
    min_y_val = jnp.minimum(min_y_candidate, max_y_val - 1e-9) # Ensure min_y < max_y

    output_common_params = {
        'type_index': all_possible_names.index(FUNCTION_NAME),
        'optimum_point': optimum_point,
        'max_y': max_y_val,
        'min_y': min_y_val,
        'action_dim': dim,
        'bounds': tuple((lower, upper)), # Store actual bounds used
    }

    # Start with a copy of the full specific_configs template
    output_specific_params = jax.tree_util.tree_map(lambda x: x, env_params_instance.sampler_configs['specific'])
    
    # Update only Gaussian's specific sampled parameters
    output_specific_params[FUNCTION_NAME]['center'] = center
    output_specific_params[FUNCTION_NAME]['width'] = width
    output_specific_params[FUNCTION_NAME]['amplitude'] = amplitude
    output_specific_params[FUNCTION_NAME]['baseline'] = baseline
    # Other params in output_specific_params[FUNCTION_NAME] (like bounds, factors) remain as configured in template.

    return {'common': output_common_params, 'specific': output_specific_params}

def compute_y_func(x: chex.Array, sampler_params: Dict, env_params_instance: 'EnvParams') -> chex.Array:
    """Computes Gaussian function value."""
    # common_params = sampler_params['common'] # Not directly needed if specific params hold all
    gaussian_specific_params = sampler_params['specific'][FUNCTION_NAME]
    
    center = gaussian_specific_params['center']
    width = gaussian_specific_params['width']
    amplitude = gaussian_specific_params['amplitude']
    baseline = gaussian_specific_params['baseline']

    x_eval = x[jnp.newaxis, :] if x.ndim == 1 else x # Ensure batch, dim: (batch, action_dim)

    diff = x_eval - center # Broadcasting: (batch, dim) - (dim,) -> (batch, dim)
    # Ensure width is not zero for division
    safe_width = jnp.maximum(width, 1e-9)
    scaled_diff_sq = (diff / safe_width) ** 2
    exponent = -0.5 * jnp.sum(scaled_diff_sq, axis=-1) # Sum over dimensions -> (batch,)
    result = baseline + amplitude * jnp.exp(exponent) # Shape (batch,)

    return result.squeeze() # Squeeze if original x was 1D
