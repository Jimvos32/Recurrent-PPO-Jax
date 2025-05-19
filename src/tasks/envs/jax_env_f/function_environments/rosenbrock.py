import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..jax_disp_samplers import EnvParams

FUNCTION_NAME = "rosenbrock"

_DEFAULT_X_RANGE = (-5.0, 10.0) # Standard domain often cited
_MAX_Y_FLIPPED_THEORETICAL = 0.0 # Global max of flipped Rosenbrock is 0 at (1,1,...,1)

def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    """Provides the specific config template for the Rosenbrock function."""
    f_env_config_key = f"{FUNCTION_NAME}_env"
    func_specific_run_config = run_config.get(f_env_config_key, {})
    
    if action_dim < 1:
        pass # Config validation should catch this.

    template = {
        'bounds': tuple(func_specific_run_config.get("bounds", _DEFAULT_X_RANGE)),
        # Rosenbrock has no other specific tunable parameters beyond its definition.
        'fixed_max_y': func_specific_run_config.get("fixed_max_y", _MAX_Y_FLIPPED_THEORETICAL),
        'fixed_min_y': func_specific_run_config.get("fixed_min_y", None) # Estimate if None
    }
    if template['fixed_max_y'] is not None:
        template['fixed_max_y'] = jnp.array(template['fixed_max_y'], dtype=jnp.float64)
    if template['fixed_min_y'] is not None:
        template['fixed_min_y'] = jnp.array(template['fixed_min_y'], dtype=jnp.float64)
    return template

def initialize_func(key: chex.PRNGKey,
                    env_params_instance: 'EnvParams',
                    action_dim: int,
                    all_possible_names: List[str]) -> Dict:
    """Initializes Rosenbrock function parameters."""
    rosenbrock_base_config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]
    dim = action_dim
    lower, upper = rosenbrock_base_config['bounds']

    # Theoretical optimum for flipped Rosenbrock is (1,1,...,1) with value 0.
    optimum_point_val = jnp.ones(dim, dtype=jnp.float64)
    
    # Check if theoretical optimum is within current bounds
    if not (jnp.all(optimum_point_val >= lower) and jnp.all(optimum_point_val <= upper)):
        # If (1,...,1) is out of bounds, the max within bounds is likely at a corner/boundary
        # closest to (1,...,1). For simplicity, we can estimate or use a fallback.
        # Fallback: use center of the domain if true optimum is out of specified bounds.
        # A more robust approach would be to evaluate at corners.
        print(f"Warning: {FUNCTION_NAME} theoretical optimum (1,...,1) is outside specified bounds. Estimating optimum within bounds.")
        # We'll refine optimum_point_val based on sampled points later.
        optimum_point_val = jnp.array([(lower + upper) / 2.0] * dim, dtype=jnp.float64) # Initial guess

    max_y_val = rosenbrock_base_config.get('fixed_max_y', _MAX_Y_FLIPPED_THEORETICAL)
    min_y_val = rosenbrock_base_config.get('fixed_min_y')

    if min_y_val is None or not (jnp.all(jnp.ones(dim) >= lower) and jnp.all(jnp.ones(dim) <= upper)):
        # Estimate min_y (and refine max_y/optimum_point) if min_y not fixed or optimum out of bounds
        num_est_samples = 100 * dim if dim <= 4 else 400 + (dim-4)*20 # Cap samples
        rand_key, _ = jax.random.split(key)
        test_points_random = jax.random.uniform(rand_key, shape=(num_est_samples, dim),
                                                minval=lower, maxval=upper, dtype=jnp.float64)
        all_test_points_list = [test_points_random]
        if dim <= 6: # Limit corners
            corners = []
            for i in range(1 << dim):
                corner = jnp.array([upper if (i >> j) & 1 else lower for j in range(dim)], dtype=jnp.float64)
                corners.append(corner)
            if corners: all_test_points_list.append(jnp.stack(corners))
        
        # Add theoretical optimum if it was within original bounds, or current best guess
        if jnp.all(jnp.ones(dim) >= lower) and jnp.all(jnp.ones(dim) <= upper):
             all_test_points_list.append(jnp.ones((1,dim), dtype=jnp.float64))
        else:
             all_test_points_list.append(optimum_point_val.reshape(1,-1))


        test_points = jnp.concatenate(all_test_points_list, axis=0)
        
        temp_sampler_params = {'common': {'action_dim': dim}, 'specific': {FUNCTION_NAME: {}}}
        y_on_samples = compute_y_func(test_points, temp_sampler_params, env_params_instance)

        current_max_y_on_samples = jnp.max(y_on_samples)
        current_min_y_on_samples = jnp.min(y_on_samples)
        opt_idx_on_samples = jnp.argmax(y_on_samples)
        
        max_y_val = jnp.maximum(max_y_val, current_max_y_on_samples) # Respect fixed_max_y but update if sample is better
        optimum_point_val = test_points[opt_idx_on_samples] # Update optimum based on samples

        if min_y_val is None: min_y_val = current_min_y_on_samples
        else: min_y_val = jnp.minimum(min_y_val, current_min_y_on_samples)

    max_y_val = jnp.minimum(max_y_val, _MAX_Y_FLIPPED_THEORETICAL) # Cap at theoretical max
    min_y_val = jnp.minimum(min_y_val, max_y_val - 1e-9)

    output_common_params = {
        'type_index': all_possible_names.index(FUNCTION_NAME),
        'optimum_point': optimum_point_val.astype(jnp.float64),
        'max_y': jnp.array(max_y_val, dtype=jnp.float64),
        'min_y': jnp.array(min_y_val, dtype=jnp.float64),
        'action_dim': dim,
        'bounds': tuple((float(lower), float(upper))),
    }
    output_specific_params = jax.tree_util.tree_map(lambda x: x, env_params_instance.sampler_configs['specific'])
    # Rosenbrock has no specific *sampled* parameters.

    return {'common': output_common_params, 'specific': output_specific_params}

def compute_y_func(x: chex.Array, sampler_params: Dict, env_params_instance: 'EnvParams') -> chex.Array:
    """
    Computes Flipped Rosenbrock function value.
    Original (minimization): sum_{i=1}^{N-1} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    This (maximization): - sum_{i=1}^{N-1} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    For N=1, original is (x_1-1)^2. Flipped is -(x_1-1)^2.
    """
    dim = sampler_params['common']['action_dim']

    x_eval = x[jnp.newaxis, :] if x.ndim == 1 else x
    chex.assert_shape(x_eval, (None, dim))

    if dim == 1:
        # Original: (x_0 - 1)^2. Flipped for maximization: -(x_0 - 1)^2
        # Note: your old code had (1 - x0)^2 which is the same.
        f_val = (x_eval[:, 0] - 1.0)**2
    else:
        # x_i terms (from x_0 to x_{N-2})
        xi = x_eval[:, :-1]       # Shape (batch, dim-1)
        # x_{i+1} terms (from x_1 to x_{N-1})
        xi_plus_1 = x_eval[:, 1:] # Shape (batch, dim-1)
        
        term1 = 100.0 * (xi_plus_1 - xi**2)**2
        term2 = (xi - 1.0)**2 # Note: (x_i - 1)^2 is standard, not (1 - x_i)^2, though square is same.
        f_val = jnp.sum(term1 + term2, axis=1) # Sum over (dim-1) terms -> (batch,)
        
    y = -f_val # Flipping for maximization
    return y.squeeze()
