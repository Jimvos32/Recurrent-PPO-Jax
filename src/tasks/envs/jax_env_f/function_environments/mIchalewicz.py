import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..jax_disp_samplers import EnvParams # Adjust path as per your final structure

FUNCTION_NAME = "michalewicz"

_DEFAULT_X_RANGE = (0.0, jnp.pi) # Standard domain [0, pi]
_DEFAULT_M_PARAM = 10.0         # Standard m parameter

# Approximate known global MINIMA of the original Michalewicz function for m=10.
# We want to MAXIMIZE, so the negative of these are our target max_y values.
_KNOWN_OPTIMA_ORIGINAL_MIN_VALUE = {
    1: -0.99656, # x1 approx 2.203
    2: -1.9662,  # (x1,x2) approx (2.203, 1.570) i.e. (2.203, pi/2)
    # For higher dimensions, the exact location is complex.
    # Values from: https://www.sfu.ca/~ssurjano/michal.html
    5: -4.687658,
    10: -9.66015
}

def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    """Provides the specific config template for the Michalewicz function."""
    f_env_config_key = f"{FUNCTION_NAME}_env"
    func_specific_run_config = run_config.get(f_env_config_key, {})

    if action_dim < 1:
        # This should ideally be caught by config validation before this point.
        # For robustness, allow template generation but functions might error later.
        pass

    template = {
        'm_param': func_specific_run_config.get("m", _DEFAULT_M_PARAM),
        'bounds': tuple(func_specific_run_config.get("bounds", _DEFAULT_X_RANGE)),
    }

    # Add fixed_max_y if known for this dim and m_param is standard
    known_opt_val = _KNOWN_OPTIMA_ORIGINAL_MIN_VALUE.get(action_dim)
    if known_opt_val is not None and jnp.isclose(jnp.array(template['m_param'], dtype=jnp.float64), _DEFAULT_M_PARAM):
        template['fixed_max_y'] = -known_opt_val # Maximize the flipped function
    else:
        template['fixed_max_y'] = None # Needs estimation

    template['fixed_min_y'] = None # Always estimate min_y as it's sensitive to bounds/dim

    # Ensure parameters are JAX arrays with correct dtype
    template['m_param'] = jnp.array(template['m_param'], dtype=jnp.float64)
    if template['fixed_max_y'] is not None:
        template['fixed_max_y'] = jnp.array(template['fixed_max_y'], dtype=jnp.float64)

    return template

def initialize_func(key: chex.PRNGKey,
                    env_params_instance: 'EnvParams',
                    action_dim: int,
                    all_possible_names: List[str]) -> Dict:
    """Initializes Michalewicz function parameters."""
    michalewicz_base_config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]
    dim = action_dim
    lower, upper = michalewicz_base_config['bounds']
    # m_param is fixed from config, not sampled here.

    max_y_val = michalewicz_base_config.get('fixed_max_y')
    min_y_val = michalewicz_base_config.get('fixed_min_y') # Usually None

    optimum_point_val = None

    # Estimation for optimum_point, and min_y/max_y if not fixed
    # Use a small number of random samples and corners for JAX-friendly estimation
    num_est_samples_per_dim_factor = 20 # Factor to multiply by dim for samples
    num_est_samples = num_est_samples_per_dim_factor * dim
    if dim > 5: # Cap samples for very high dimensions
        num_est_samples = num_est_samples_per_dim_factor * 5 + (dim - 5) * 5

    rand_key, key_corners = jax.random.split(key)
    test_points_random = jax.random.uniform(rand_key, shape=(num_est_samples, dim),
                                            minval=lower, maxval=upper, dtype=jnp.float64)
    
    all_test_points_list = [test_points_random]

    # Add corners (be mindful of 2^dim complexity for high dims)
    if dim <= 6: # Limit corner evaluation for higher dimensions
        corners = []
        for i in range(1 << dim):
            corner_point = jnp.array([upper if (i >> j) & 1 else lower for j in range(dim)], dtype=jnp.float64)
            corners.append(corner_point)
        if corners:
            all_test_points_list.append(jnp.stack(corners))
    
    test_points = jnp.concatenate(all_test_points_list, axis=0)

    # Temporarily create sampler_params for compute_y_func call during init
    # Pass m_param which is needed for computation.
    temp_sampler_params = {
        'common': {'action_dim': dim},
        'specific': {FUNCTION_NAME: {'m_param': michalewicz_base_config['m_param']}}
    }
    y_on_samples = compute_y_func(test_points, temp_sampler_params, env_params_instance)

    current_max_y_on_samples = jnp.max(y_on_samples)
    current_min_y_on_samples = jnp.min(y_on_samples)
    opt_idx_on_samples = jnp.argmax(y_on_samples)
    optimum_point_on_samples = test_points[opt_idx_on_samples]

    if max_y_val is None:
        max_y_val = current_max_y_on_samples
    else: # If fixed_max_y was provided, ensure it's at least as good as sampled
        max_y_val = jnp.maximum(max_y_val, current_max_y_on_samples)

    if min_y_val is None:
        min_y_val = current_min_y_on_samples
    else: # If fixed_min_y was provided (though typically not for Michalewicz)
        min_y_val = jnp.minimum(min_y_val, current_min_y_on_samples)
        
    min_y_val = jnp.minimum(min_y_val, max_y_val - 1e-9) # Ensure min_y < max_y
    optimum_point_val = optimum_point_on_samples # Best found from sampling

    output_common_params = {
        'type_index': all_possible_names.index(FUNCTION_NAME),
        'optimum_point': optimum_point_val.astype(jnp.float64),
        'max_y': jnp.array(max_y_val, dtype=jnp.float64),
        'min_y': jnp.array(min_y_val, dtype=jnp.float64),
        'action_dim': dim,
        'bounds': tuple((float(lower), float(upper))), # Store as Python floats for consistency
    }
    # The 'specific' part of the output for Michalewicz only contains 'm_param' from the template.
    output_specific_params = jax.tree_util.tree_map(lambda x: x, env_params_instance.sampler_configs['specific'])

    return {'common': output_common_params, 'specific': output_specific_params}

def compute_y_func(x: chex.Array, sampler_params: Dict, env_params_instance: 'EnvParams') -> chex.Array:
    """
    Computes the Michalewicz function value (formulated for maximization).
    Original (minimization): - sum_{i=1 to d} sin(x_i) * [sin(i * x_i^2 / pi)]^(2m)
    This implementation (maximization): sum_{i=1 to d} sin(x_i) * [sin(i * x_i^2 / pi)]^(2m)
    """
    michalewicz_specific_params = sampler_params['specific'][FUNCTION_NAME]
    m_param = michalewicz_specific_params['m_param'] # Fixed m from config
    dim = sampler_params['common']['action_dim']

    x_eval = x[jnp.newaxis, :] if x.ndim == 1 else x # Ensure (batch, dim)
    chex.assert_shape(x_eval, (None, dim))

    # Create an array for i = 1, 2, ..., dim
    i_values = jnp.arange(1, dim + 1, dtype=jnp.float64).reshape(1, -1) # Shape (1, dim)

    term_in_sin_exponent = i_values * (x_eval**2) / jnp.pi # Shape (batch, dim)
    
    # sin(term_in_sin_exponent)^(2m)
    sin_val_exponent_base = jnp.sin(term_in_sin_exponent)
    
    # Power for JAX: jnp.power(base, exp).
    # Ensure base is non-negative if exp is fractional. Here 2*m is usually an even integer.
    # Add a small epsilon to the base if sin_val_exponent_base can be negative and 2*m is not perfectly even due to float m.
    # However, (sin(y))^even_power is equivalent to (|sin(y)|)^even_power.
    # A safer approach for general m (even if m=10.0 is typical):
    # sin_term_pow_2m = jnp.power(jnp.abs(sin_val_exponent_base) + 1e-9, 2.0 * m_param)
    # If m is guaranteed to make 2*m an even integer, direct power is fine.
    # For m=10, 2*m = 20.
    sin_term_pow_2m = jnp.power(sin_val_exponent_base, 2.0 * m_param)
    
    term_i = jnp.sin(x_eval) * sin_term_pow_2m # Element-wise, shape (batch, dim)
    
    total_y_sum = jnp.sum(term_i, axis=1) # Sum over dimensions -> shape (batch,)
    
    return total_y_sum.squeeze() # Squeeze if original x was 1D (batch=1)
