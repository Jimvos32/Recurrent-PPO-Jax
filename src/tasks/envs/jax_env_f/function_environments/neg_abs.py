import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..jax_disp_samplers import EnvParams

FUNCTION_NAME = "neg_abs"

def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    """Provides the specific config template for the Negative Absolute Value function."""
    f_env_config_key = f"{FUNCTION_NAME}_env"
    func_specific_run_config = run_config.get(f_env_config_key, {})
    default_bounds = run_config.get("bounds", (-5.0, 5.0))

    # neg_abs typically doesn't have its own tunable parameters beyond bounds
    return {
        "bounds": tuple(func_specific_run_config.get("bounds", default_bounds)),
    }

def initialize_func(key: chex.PRNGKey,
                    env_params_instance: 'EnvParams',
                    action_dim: int,
                    all_possible_names: List[str]) -> Dict:
    """Initializes Negative Absolute Value function parameters."""
    neg_abs_base_config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]
    dim = action_dim

    lower, upper = neg_abs_base_config['bounds']

    optimum_point = jnp.zeros((dim,), dtype=jnp.float64)
    max_y_val = 0.0 # Max of -sum(|x_i|) is 0 at x_i = 0

    # Min value occurs at the corners furthest from the origin.
    jnp_lower = jnp.array(lower, dtype=jnp.float64)
    jnp_upper = jnp.array(upper, dtype=jnp.float64)
    corner_coord_abs_max = jnp.maximum(jnp.abs(jnp_lower), jnp.abs(jnp_upper))
    min_y_candidate = -jnp.sum(jnp.full((dim,), corner_coord_abs_max))
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
    # neg_abs has no specific *sampled* parameters to store in its own specific dict entry.
    # output_specific_params[FUNCTION_NAME] will remain as defined by its template (e.g., just containing 'bounds').

    return {'common': output_common_params, 'specific': output_specific_params}

def compute_y_func(x: chex.Array, sampler_params: Dict, env_params_instance: 'EnvParams') -> chex.Array:
    """Computes Negative Absolute Value function f(x) = -sum(|x_i|)."""
    # No specific parameters needed from sampler_params['specific']['neg_abs'] for computation
    
    x_eval = x[jnp.newaxis, :] if x.ndim == 1 else x # Ensure (batch, dim)

    result = -jnp.sum(jnp.abs(x_eval), axis=-1) # Sum over dimensions -> (batch,)
    
    return result.squeeze() # Squeeze if original x was 1D
