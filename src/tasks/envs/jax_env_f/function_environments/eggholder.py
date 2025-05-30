import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..jax_disp_samplers import EnvParams

FUNCTION_NAME = "eggholder"

_DEFAULT_X_RANGE = (-512.0, 512.0)
# Original Eggholder min is approx -959.6407 at (512, 404.2319) for 2D.
# Flipped max_y for 2D is approx 959.6407.
_FIXED_MAX_Y_FLIPPED_2D_STD_BOUNDS = 959.6407
_OPTIMUM_POINT_2D_STD_BOUNDS = jnp.array([512.0, 404.2319])
# Min_y for N-D is harder; your previous code used a general -1049.0.

def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    """Provides the specific config template for the Eggholder function."""
    f_env_config_key = f"{FUNCTION_NAME}_env"
    func_specific_run_config = run_config.get(f_env_config_key, {})

    if action_dim < 2:
        # Eggholder is defined for action_dim >= 2.
        pass

    template = {
        'bounds': tuple(func_specific_run_config.get("bounds", _DEFAULT_X_RANGE)),
    }
    # Use fixed max_y if 2D and standard bounds, otherwise estimate
    if action_dim == 2 and jnp.allclose(jnp.array(template['bounds']), jnp.array(_DEFAULT_X_RANGE)):
        template['fixed_max_y'] = func_specific_run_config.get("fixed_max_y", _FIXED_MAX_Y_FLIPPED_2D_STD_BOUNDS)
    else:
        template['fixed_max_y'] = func_specific_run_config.get("fixed_max_y", None) # Estimate for N-D or non-standard bounds

    # Use fixed min_y from config if provided, otherwise estimate
    template['fixed_min_y'] = func_specific_run_config.get("fixed_min_y", None) # Will be estimated if None

    if template['fixed_max_y'] is not None:
        template['fixed_max_y'] = jnp.array(template['fixed_max_y'], dtype=jnp.float64)
    if template['fixed_min_y'] is not None:
        template['fixed_min_y'] = jnp.array(template['fixed_min_y'], dtype=jnp.float64)
    return template

def initialize_func(key: chex.PRNGKey,
                    env_params_instance: 'EnvParams',
                    action_dim: int,
                    all_possible_names: List[str]) -> Dict:
    """Initializes Eggholder function parameters."""
    eggholder_base_config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]
    dim = action_dim
    
    if dim < 2:
      raise ValueError(f"{FUNCTION_NAME} function requires action_dim >= 2, got {dim}.")

    lower, upper = eggholder_base_config['bounds']

    max_y_val = eggholder_base_config.get('fixed_max_y')
    min_y_val = eggholder_base_config.get('fixed_min_y')
    optimum_point_val = None

    if dim == 2 and max_y_val is not None and \
       jnp.allclose(jnp.array(eggholder_base_config['bounds']), jnp.array(_DEFAULT_X_RANGE)):
        optimum_point_val = _OPTIMUM_POINT_2D_STD_BOUNDS

    # If values are not fixed or optimum point not set, estimate them
    if optimum_point_val is None or max_y_val is None or min_y_val is None:
        num_est_samples_per_dim_factor = 15
        num_est_samples = num_est_samples_per_dim_factor * dim
        if dim > 4: # Cap samples for higher dim
            num_est_samples = num_est_samples_per_dim_factor * 4 + (dim - 4) * 5
        
        rand_key, key_corners = jax.random.split(key)
        test_points_random = jax.random.uniform(rand_key, shape=(num_est_samples, dim),
                                                minval=lower, maxval=upper, dtype=jnp.float64)
        all_test_points_list = [test_points_random]

        if dim <= 5: # Limit corner evaluation
            corners = []
            for i in range(1 << dim):
                corner = jnp.array([upper if (i >> j) & 1 else lower for j in range(dim)], dtype=jnp.float64)
                corners.append(corner)
            if corners:
                all_test_points_list.append(jnp.stack(corners))
        
        # If 2D and standard optimum known, add it to test points
        if dim == 2 and jnp.allclose(jnp.array(eggholder_base_config['bounds']), jnp.array(_DEFAULT_X_RANGE)):
             all_test_points_list.append(_OPTIMUM_POINT_2D_STD_BOUNDS.reshape(1,-1))

        test_points = jnp.concatenate(all_test_points_list, axis=0)
        
        # Minimal sampler_params for compute_y_func during init (Eggholder has no specific sampled params)
        temp_sampler_params = {'common': {'action_dim': dim}, 'specific': {FUNCTION_NAME: {}}}
        y_on_samples = compute_y_func(test_points, temp_sampler_params, env_params_instance)

        current_max_y_on_samples = jnp.max(y_on_samples)
        current_min_y_on_samples = jnp.min(y_on_samples)
        opt_idx_on_samples = jnp.argmax(y_on_samples)
        
        if max_y_val is None: max_y_val = current_max_y_on_samples
        else: max_y_val = jnp.maximum(max_y_val, current_max_y_on_samples)
        
        if min_y_val is None: min_y_val = current_min_y_on_samples
        else: min_y_val = jnp.minimum(min_y_val, current_min_y_on_samples)

        if optimum_point_val is None: # If not the standard 2D case with known optimum
            optimum_point_val = test_points[opt_idx_on_samples]
    
    min_y_val = jnp.minimum(min_y_val, max_y_val - 1e-9)

    output_common_params = {
        'type_index': all_possible_names.index(FUNCTION_NAME),
        'optimum_point': optimum_point_val.astype(jnp.float64),
        'max_y': jnp.array(max_y_val, dtype=jnp.float64),
        'min_y': jnp.array(min_y_val, dtype=jnp.float64),
        'action_dim': dim,
        'bounds': tuple((lower, upper)),
    }
    output_specific_params = jax.tree_util.tree_map(lambda x: x, env_params_instance.sampler_configs['specific'])
    # Eggholder has no specific *sampled* parameters to store.

    return {'common': output_common_params, 'specific': output_specific_params}

def compute_y_func(x: chex.Array, sampler_params: Dict, env_params_instance: 'EnvParams') -> chex.Array:
    """
    Computes Flipped N-dimensional Eggholder function value.
    Original formula: sum_{i=0}^{N-2} [ -(x_{i+1} + 47) sin(sqrt(abs(x_i/2 + (x_{i+1} + 47))))
                                       - x_i sin(sqrt(abs(x_i - (x_{i+1} + 47)))) ]
    This returns:       sum_{i=0}^{N-2} [  (x_{i+1} + 47) sin(sqrt(abs(x_i/2 + (x_{i+1} + 47))))
                                       + x_i sin(sqrt(abs(x_i - (x_{i+1} + 47)))) ]
    So it's maximized.
    """
    dim = sampler_params['common']['action_dim'] # Or env_params_instance.action_dim

    x_eval = x[jnp.newaxis, :] if x.ndim == 1 else x
    chex.assert_shape(x_eval, (None, dim))

    total_y_sum = jnp.zeros(x_eval.shape[0], dtype=jnp.float64)

    # Sum over pairs (x_i, x_{i+1})
    # The loop runs dim-1 times.
    for i in range(dim - 1):
        xi = x_eval[:, i]
        xi_plus_1 = x_eval[:, i + 1]

        # Add a small epsilon inside sqrt to prevent NaN gradients if AD is ever used,
        # and to ensure sqrt argument is non-negative for robustness.
        epsilon = 1e-9 

        term1_sqrt_arg = jnp.abs(xi / 2.0 + (xi_plus_1 + 47.0)) + epsilon
        term1 = -(xi_plus_1 + 47.0) * jnp.sin(jnp.sqrt(term1_sqrt_arg))

        term2_sqrt_arg = jnp.abs(xi - (xi_plus_1 + 47.0)) + epsilon
        term2 = -xi * jnp.sin(jnp.sqrt(term2_sqrt_arg))
        
        total_y_sum += (term1 + term2) # This is the original Eggholder sum
    
    # Flipping the original sum for maximization
    y = -total_y_sum 
    return y.squeeze()
