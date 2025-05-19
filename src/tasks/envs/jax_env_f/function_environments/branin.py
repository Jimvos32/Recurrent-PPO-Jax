import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..jax_disp_samplers import EnvParams

FUNCTION_NAME = "branin"

# Standard Branin parameters (as JAX arrays for consistency)
_A_STD = jnp.array(1.0, dtype=jnp.float64)
_B_STD = jnp.array(5.1 / (4.0 * jnp.pi**2), dtype=jnp.float64)
_C_STD = jnp.array(5.0 / jnp.pi, dtype=jnp.float64)
_R_STD = jnp.array(6.0, dtype=jnp.float64)
_S_STD = jnp.array(10.0, dtype=jnp.float64)
_T_STD = jnp.array(1.0 / (8.0 * jnp.pi), dtype=jnp.float64)

_OPTIMUM_VALUE_ORIGINAL_BRANIN = 0.397887 # Approx min value of original Branin
_MAX_Y_FLIPPED_BRANIN = -_OPTIMUM_VALUE_ORIGINAL_BRANIN

_DEFAULT_X_RANGES_BRANIN = [(-5.0, 10.0), (0.0, 15.0)] # (x1_min, x1_max), (x2_min, x2_max)
_OPTIMA_LOCATIONS_ORIGINAL_BRANIN = jnp.array([
    [-jnp.pi, 12.275],
    [jnp.pi, 2.275],
    [3 * jnp.pi, 2.475] # Approx 9.42478
], dtype=jnp.float64)


def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    """Provides the specific config template for the Branin function (strictly 2D)."""
    if action_dim != 2:
      # This should be an error or handled upstream by config validation if Branin is selected.
      # For template generation, proceed, but it's for a 2D function.
      pass
    
    f_env_config_key = f"{FUNCTION_NAME}_env"
    func_specific_run_config = run_config.get(f_env_config_key, {})

    template = {
        'a': jnp.array(func_specific_run_config.get("a", _A_STD), dtype=jnp.float64),
        'b': jnp.array(func_specific_run_config.get("b", _B_STD), dtype=jnp.float64),
        'c': jnp.array(func_specific_run_config.get("c", _C_STD), dtype=jnp.float64),
        'r': jnp.array(func_specific_run_config.get("r", _R_STD), dtype=jnp.float64),
        's': jnp.array(func_specific_run_config.get("s", _S_STD), dtype=jnp.float64),
        't': jnp.array(func_specific_run_config.get("t", _T_STD), dtype=jnp.float64),
        'bounds': [ # List of two tuples for x1 and x2 bounds
            tuple(func_specific_run_config.get("bounds_x1", _DEFAULT_X_RANGES_BRANIN[0])),
            tuple(func_specific_run_config.get("bounds_x2", _DEFAULT_X_RANGES_BRANIN[1]))
        ],
        'fixed_max_y': jnp.array(func_specific_run_config.get("fixed_max_y", _MAX_Y_FLIPPED_BRANIN), dtype=jnp.float64),
        'fixed_min_y': func_specific_run_config.get("fixed_min_y", None) # Estimate if None
    }
    if template['fixed_min_y'] is not None:
        template['fixed_min_y'] = jnp.array(template['fixed_min_y'], dtype=jnp.float64)
    return template

def initialize_func(key: chex.PRNGKey,
                    env_params_instance: 'EnvParams',
                    action_dim: int,
                    all_possible_names: List[str]) -> Dict:
    """Initializes Branin function parameters (fixed 2D)."""
    if action_dim != 2:
      raise ValueError(f"{FUNCTION_NAME} function requires action_dim = 2, got {action_dim}.")
    dim = 2

    branin_base_config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]
    
    bounds_x1_tuple = branin_base_config['bounds'][0]
    bounds_x2_tuple = branin_base_config['bounds'][1]

    max_y_val = branin_base_config.get('fixed_max_y', _MAX_Y_FLIPPED_BRANIN)
    min_y_val = branin_base_config.get('fixed_min_y')
    
    # Determine optimum point: one of the known optima if within current bounds
    optimum_point_val = None
    best_opt_y_in_bounds = -jnp.inf # For flipped function, higher is better

    for loc in _OPTIMA_LOCATIONS_ORIGINAL_BRANIN:
        # Check if loc is within [bounds_x1_tuple, bounds_x2_tuple]
        if (bounds_x1_tuple[0] <= loc[0] <= bounds_x1_tuple[1]) and \
           (bounds_x2_tuple[0] <= loc[1] <= bounds_x2_tuple[1]):
            # If this known optimum is within bounds, its value is _MAX_Y_FLIPPED_BRANIN
            # (assuming standard a,b,c... params)
            # This logic assumes standard params are used if we pick a standard optimum point.
            # If params a,b,c.. are changed, the optima locations and values also change.
            # For simplicity, if using standard optima, assume standard params.
            if _MAX_Y_FLIPPED_BRANIN > best_opt_y_in_bounds: # Should only happen once if fixed_max_y is used
                 best_opt_y_in_bounds = _MAX_Y_FLIPPED_BRANIN
                 optimum_point_val = loc

    # If no standard optimum is in bounds, or if params are non-standard, estimate.
    if optimum_point_val is None or not jnp.isclose(max_y_val, _MAX_Y_FLIPPED_BRANIN):
        num_grid_samples = 40 # Per dimension for estimation (40x40 grid)
        x1_samples = jnp.linspace(bounds_x1_tuple[0], bounds_x1_tuple[1], num_grid_samples, dtype=jnp.float64)
        x2_samples = jnp.linspace(bounds_x2_tuple[0], bounds_x2_tuple[1], num_grid_samples, dtype=jnp.float64)
        grid_x1, grid_x2 = jnp.meshgrid(x1_samples, x2_samples)
        test_points = jnp.stack([grid_x1.ravel(), grid_x2.ravel()], axis=-1)

        temp_sampler_params = {
            'common': {'action_dim': dim},
            'specific': {FUNCTION_NAME: branin_base_config} # Pass fixed a,b,c...
        }
        y_on_grid = compute_y_func(test_points, temp_sampler_params, env_params_instance)
        
        current_max_y_on_grid = jnp.max(y_on_grid)
        opt_idx_on_grid = jnp.argmax(y_on_grid)
        
        if current_max_y_on_grid > best_opt_y_in_bounds : # If grid found better than known optima in bounds
            max_y_val = current_max_y_on_grid
            optimum_point_val = test_points[opt_idx_on_grid]
        elif optimum_point_val is None: # If no known optima were in bounds at all
            max_y_val = current_max_y_on_grid
            optimum_point_val = test_points[opt_idx_on_grid]
        # else, optimum_point_val is already set to a known good one, and max_y_val is its value.

        if min_y_val is None:
            min_y_val = jnp.min(y_on_grid)
        else:
            min_y_val = jnp.minimum(min_y_val, jnp.min(y_on_grid))

    if optimum_point_val is None: # Fallback if all else fails (e.g. very narrow bounds)
        optimum_point_val = jnp.array([(bounds_x1_tuple[0]+bounds_x1_tuple[1])/2, (bounds_x2_tuple[0]+bounds_x2_tuple[1])/2], dtype=jnp.float64)
        temp_sampler_params = {'common': {'action_dim': dim}, 'specific': {FUNCTION_NAME: branin_base_config}}
        max_y_val = compute_y_func(optimum_point_val.reshape(1,-1), temp_sampler_params, env_params_instance).squeeze()
        if min_y_val is None: min_y_val = max_y_val - 1.0 # Arbitrary if no other info

    min_y_val = jnp.minimum(min_y_val, max_y_val - 1e-9)

    output_common_params = {
        'type_index': all_possible_names.index(FUNCTION_NAME),
        'optimum_point': optimum_point_val.astype(jnp.float64),
        'max_y': jnp.array(max_y_val, dtype=jnp.float64),
        'min_y': jnp.array(min_y_val, dtype=jnp.float64),
        'action_dim': dim,
        'bounds': [tuple(map(float,b)) for b in branin_base_config['bounds']], # List of Python float tuples
    }
    output_specific_params = jax.tree_util.tree_map(lambda x: x, env_params_instance.sampler_configs['specific'])
    # Branin's a,b,c.. are fixed from config.

    return {'common': output_common_params, 'specific': output_specific_params}

def compute_y_func(x: chex.Array, sampler_params: Dict, env_params_instance: 'EnvParams') -> chex.Array:
    """
    Computes Flipped Branin function value. Expects x to be (batch, 2).
    Original: a(x2 - b*x1^2 + c*x1 - r)^2 + s(1-t)cos(x1) + s
    Flipped: -[a(x2 - b*x1^2 + c*x1 - r)^2 + s(1-t)cos(x1) + s]
    """
    branin_specific_params = sampler_params['specific'][FUNCTION_NAME]
    a = branin_specific_params['a']
    b = branin_specific_params['b']
    c = branin_specific_params['c']
    r = branin_specific_params['r']
    s = branin_specific_params['s']
    t = branin_specific_params['t']

    x_eval = x[jnp.newaxis, :] if x.ndim == 1 else x
    chex.assert_shape(x_eval, (None, 2))

    x1 = x_eval[:, 0]
    x2 = x_eval[:, 1]

    term_in_paren = x2 - b * x1**2 + c * x1 - r
    term1 = a * term_in_paren**2
    term2 = s * (1.0 - t) * jnp.cos(x1)
    term3 = s

    original_branin_val = term1 + term2 + term3
    y = -original_branin_val # Flipping for maximization
    
    return y.squeeze()
