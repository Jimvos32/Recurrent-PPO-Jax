import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..jax_disp_samplers import EnvParams

FUNCTION_NAME = "hartmann6" # This module is specifically for Hartmann6

# Standard parameters for Hartmann 6D
_ALPHA_STD_H6 = jnp.array([1.0, 1.2, 3.0, 3.2])
_A_MATRIX_STD_H6 = jnp.array([
    [10, 3, 17, 3.5, 1.7, 8],
    [0.05, 10, 17, 0.1, 8, 14],
    [3, 3.5, 1.7, 10, 17, 8],
    [17, 8, 0.05, 10, 0.1, 14]
])
_P_MATRIX_STD_H6 = 1e-4 * jnp.array([
    [1312, 1696, 5569, 124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091, 381]
])

_DEFAULT_X_RANGE_H6 = (0.0, 1.0)
_OPTIMUM_LOC_H6 = jnp.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
_OPTIMUM_VALUE_ORIGINAL_H6 = -3.322368 # Min value of original Hartmann6
_MAX_Y_FLIPPED_H6 = -_OPTIMUM_VALUE_ORIGINAL_H6 # Approx 3.322368 for maximization

def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    """Provides the specific config template for the Hartmann 6D function."""
    if action_dim != 6:
        # Hartmann6 is strictly 6D. This should be enforced by config or raise error.
        # For template generation, we'll assume action_dim will be 6 if this function is active.
        pass 
    
    f_env_config_key = f"{FUNCTION_NAME}_env" # e.g., "hartmann6_env"
    func_specific_run_config = run_config.get(f_env_config_key, {})

    template = {
        'alpha': func_specific_run_config.get("alpha", _ALPHA_STD_H6),
        'A_matrix': func_specific_run_config.get("A_matrix", _A_MATRIX_STD_H6),
        'P_matrix': func_specific_run_config.get("P_matrix", _P_MATRIX_STD_H6),
        'bounds': tuple(func_specific_run_config.get("bounds", _DEFAULT_X_RANGE_H6)),
        'fixed_max_y': func_specific_run_config.get("fixed_max_y", _MAX_Y_FLIPPED_H6),
        'fixed_min_y': func_specific_run_config.get("fixed_min_y", None) # Estimate if None
    }
    # Ensure JAX arrays with correct dtype
    template['alpha'] = jnp.array(template['alpha'])
    template['A_matrix'] = jnp.array(template['A_matrix'])
    template['P_matrix'] = jnp.array(template['P_matrix'])

    if template['fixed_max_y'] is not None:
        template['fixed_max_y'] = jnp.array(template['fixed_max_y'])
    if template['fixed_min_y'] is not None:
        template['fixed_min_y'] = jnp.array(template['fixed_min_y'])

    # Basic shape validation if not using defaults
    if not (jnp.array_equal(template['alpha'], _ALPHA_STD_H6) and \
            jnp.array_equal(template['A_matrix'], _A_MATRIX_STD_H6) and \
            jnp.array_equal(template['P_matrix'], _P_MATRIX_STD_H6)):
        if template['alpha'].shape != (4,): raise ValueError("Hartmann6 'alpha' must have shape (4,).")
        if template['A_matrix'].shape != (4, 6): raise ValueError("Hartmann6 'A_matrix' must have shape (4, 6).")
        if template['P_matrix'].shape != (4, 6): raise ValueError("Hartmann6 'P_matrix' must have shape (4, 6).")
    return template

def initialize_func(key: chex.PRNGKey,
                    env_params_instance: 'EnvParams',
                    action_dim: int,
                    all_possible_names: List[str]) -> Dict:
    """Initializes Hartmann 6D function parameters."""
    if action_dim != 6:
      raise ValueError(f"{FUNCTION_NAME} function requires action_dim = 6, got {action_dim}.")
    dim = 6 # Hartmann6 is 6D

    hartmann_base_config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]
    lower, upper = hartmann_base_config['bounds']

    optimum_point_val = _OPTIMUM_LOC_H6 # Known optimum for H6
    # Check if known optimum is within current bounds
    if not (jnp.all(optimum_point_val >= lower) and jnp.all(optimum_point_val <= upper)):
        # If known optimum is out of bounds, this is problematic for fixed optimum.
        # For this function, usually bounds are [0,1] and optimum is within.
        # If bounds change, the true optimum might shift or this point becomes irrelevant.
        # Fallback: use center of domain if true optimum is out of specified bounds.
        print(f"Warning: {FUNCTION_NAME} known optimum is outside specified bounds. Using domain center as fallback optimum.")
        optimum_point_val = jnp.array([(lower + upper) / 2.0] * dim, dtype=jnp.float64)

    max_y_val = hartmann_base_config.get('fixed_max_y', _MAX_Y_FLIPPED_H6)
    min_y_val = hartmann_base_config.get('fixed_min_y')

    if min_y_val is None: # Estimate min_y if not provided
        # Evaluate at corners (2^6 = 64 points)
        corners = []
        for i in range(1 << dim):
            corner = jnp.array([upper if (i >> j) & 1 else lower for j in range(dim)], dtype=jnp.float64)
            corners.append(corner)
        corners_arr = jnp.stack(corners)
        
        # Add a few random points
        rand_key, _ = jax.random.split(key)
        random_points = jax.random.uniform(rand_key, shape=(50, dim), minval=lower, maxval=upper, dtype=jnp.float64)
        test_points = jnp.concatenate([corners_arr, random_points, optimum_point_val.reshape(1,-1)], axis=0)

        temp_sampler_params = {
            'common': {'action_dim': dim},
            'specific': {FUNCTION_NAME: hartmann_base_config} # Pass fixed alpha, A, P
        }
        y_on_samples = compute_y_func(test_points, temp_sampler_params, env_params_instance)
        min_y_val = jnp.min(y_on_samples)
        # Re-check max_y based on samples, respecting fixed_max_y
        max_y_val = jnp.maximum(max_y_val, jnp.max(y_on_samples))


    min_y_val = jnp.minimum(min_y_val, max_y_val - 1e-9)

    output_common_params = {
        'type_index': all_possible_names.index(FUNCTION_NAME),
        'optimum_point': optimum_point_val.astype(jnp.float64),
        'max_y': jnp.array(max_y_val, dtype=jnp.float64),
        'min_y': jnp.array(min_y_val, dtype=jnp.float64),
        'action_dim': dim,
        'bounds': tuple((lower, upper)),
    }
    # Hartmann's alpha, A, P are fixed from config, no new sampled values for its specific section.
    output_specific_params = jax.tree_util.tree_map(lambda x: x, env_params_instance.sampler_configs['specific'])

    return {'common': output_common_params, 'specific': output_specific_params}

def compute_y_func(x: chex.Array, sampler_params: Dict, env_params_instance: 'EnvParams') -> chex.Array:
    """
    Computes Flipped Hartmann 6D function value.
    Original (minimization): - sum_{i=1 to 4} [alpha_i * exp(-sum_{j=1 to 6} A_ij * (x_j - P_ij)^2)]
    This (maximization): sum_{i=1 to 4} [alpha_i * exp(-sum_{j=1 to 6} A_ij * (x_j - P_ij)^2)]
    """
    hartmann_specific_params = sampler_params['specific'][FUNCTION_NAME]
    alpha = hartmann_specific_params['alpha']     # Shape (4,)
    A_matrix = hartmann_specific_params['A_matrix'] # Shape (4, 6)
    P_matrix = hartmann_specific_params['P_matrix'] # Shape (4, 6)
    
    dim = sampler_params['common']['action_dim'] # Should be 6
    
    x_eval = x[jnp.newaxis, :] if x.ndim == 1 else x
    chex.assert_shape(x_eval, (None, dim)) # x_eval is (batch, 6)

    outer_sum_val = jnp.zeros(x_eval.shape[0], dtype=jnp.float64)

    for i in range(4): # Loop for alpha_i, A_i row, P_i row
        A_i_row = A_matrix[i, :] # Shape (6,)
        P_i_row = P_matrix[i, :] # Shape (6,)
        
        # exponent part: sum_{j=1 to 6} A_ij * (x_j - P_ij)^2
        # x_eval is (batch, 6), P_i_row is (6,).
        diff = x_eval - P_i_row.reshape(1, -1) # Shape (batch, 6)
        diff_sq = diff**2                      # Shape (batch, 6)
        
        # A_i_row is (6,).
        term_in_inner_sum = A_i_row.reshape(1, -1) * diff_sq # Shape (batch, 6)
        inner_sum_val = jnp.sum(term_in_inner_sum, axis=1)    # Sum over j, shape (batch,)
        
        outer_sum_val += alpha[i] * jnp.exp(-inner_sum_val)
        
    # The original Hartmann formula often includes a negative sign before the outer sum,
    # making it a minimization problem. To maximize, we use the positive sum.
    # Your previous sampler's compute_y returned -total_y, where total_y was the sum of positive terms.
    # So, if total_y = sum(alpha * exp(...)), then for maximization, we want this value.
    # If the standard definition is sum(alpha * exp(...)) and it's minimized, then -sum(...) is maximized.
    # The common definition of Hartmann is y(x) = - sum alpha_i exp(...).
    # So, to maximize, we want -y(x) = sum alpha_i exp(...).
    return outer_sum_val.squeeze()
