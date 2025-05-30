import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..jax_disp_samplers import EnvParams

FUNCTION_NAME = "cosine" # Or a shorter unique name like "cosineosc"

def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    """Provides the specific config template for the Cosine with Oscillations function."""
    f_env_config_key = f"{FUNCTION_NAME}_env" # Use a consistent key for config
    func_specific_run_config = run_config.get(f_env_config_key, {})
    default_bounds = run_config.get("bounds", (-5.0, 5.0)) # Global default

    num_oscillations = int(func_specific_run_config.get("num_oscillations", 3))

    return {
        'num_oscillations': num_oscillations,
        'c_val_sampled': jnp.nan, # For sampled 'c'
        'A0_sampled': jnp.nan,    # For sampled 'A0'
        'B0_sampled': jnp.full((action_dim,), jnp.nan, dtype=jnp.float64),
        's0_optimum_sampled': jnp.full((action_dim,), jnp.nan, dtype=jnp.float64),
        'small_A_sampled': jnp.full((action_dim, num_oscillations), jnp.nan, dtype=jnp.float64),
        'small_B_sampled': jnp.full((action_dim, num_oscillations), jnp.nan, dtype=jnp.float64),
        'small_shift_sampled': jnp.full((action_dim, num_oscillations), jnp.nan, dtype=jnp.float64),
        'small_phase_sampled': jnp.full((action_dim, num_oscillations), jnp.nan, dtype=jnp.float64),
        
        'bounds': tuple(func_specific_run_config.get("bounds", default_bounds)),
        'c_bounds_config': tuple(func_specific_run_config.get("c_bounds", (1.0, 5.0))),
        'A0_bounds_config': tuple(func_specific_run_config.get("A_bounds", (5.0, 10.0))),
        'B0_bounds_config': tuple(func_specific_run_config.get("B_bounds", (1.0, 2.0))),
        's0_opt_factor_config': func_specific_run_config.get("s0_opt_factor", 0.9), # Factor for optimum placement
        'small_A_bounds_config': tuple(func_specific_run_config.get("small_A_bounds", (0.2, 3.0))),
        'small_B_bounds_config': tuple(func_specific_run_config.get("small_B_bounds", (0.5, 4.0))),
    }

def initialize_func(key: chex.PRNGKey,
                    env_params_instance: 'EnvParams',
                    action_dim: int,
                    all_possible_names: List[str]) -> Dict:
    """Initializes Cosine with Oscillations function parameters."""
    cosine_base_config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]
    dim = action_dim

    key_c, key_A0, key_B0, key_s0, key_smA, key_smB, key_smP = jax.random.split(key, 7)

    num_oscillations = cosine_base_config['num_oscillations']
    lower, upper = cosine_base_config['bounds']
    s0_opt_factor = cosine_base_config['s0_opt_factor_config']

    c_val = jax.random.uniform(key_c, shape=(), minval=cosine_base_config['c_bounds_config'][0], maxval=cosine_base_config['c_bounds_config'][1], dtype=jnp.float64)
    A0_val = jax.random.uniform(key_A0, shape=(), minval=cosine_base_config['A0_bounds_config'][0], maxval=cosine_base_config['A0_bounds_config'][1], dtype=jnp.float64)
    B0_val = jax.random.uniform(key_B0, shape=(dim,), minval=cosine_base_config['B0_bounds_config'][0], maxval=cosine_base_config['B0_bounds_config'][1], dtype=jnp.float64)
    
    # Place s0 (optimum for main cosine) slightly inwards
    s0_min = lower + (upper - lower) * (1 - s0_opt_factor) / 2
    s0_max = upper - (upper - lower) * (1 - s0_opt_factor) / 2
    s0_val = jax.random.uniform(key_s0, shape=(dim,), minval=s0_min, maxval=s0_max, dtype=jnp.float64)
    optimum_point_val = s0_val # The main cosine term is maximized here

    small_A_val = jax.random.uniform(key_smA, shape=(dim, num_oscillations), minval=cosine_base_config['small_A_bounds_config'][0], maxval=cosine_base_config['small_A_bounds_config'][1], dtype=jnp.float64)
    small_B_val = jax.random.uniform(key_smB, shape=(dim, num_oscillations), minval=cosine_base_config['small_B_bounds_config'][0], maxval=cosine_base_config['small_B_bounds_config'][1], dtype=jnp.float64)
    small_shift_val = jnp.tile(s0_val[:, jnp.newaxis], (1, num_oscillations)) # Small oscillations centered around s0_val
    small_phase_val = jax.random.uniform(key_smP, shape=(dim, num_oscillations), minval=0, maxval=2*jnp.pi, dtype=jnp.float64)


    # Theoretical bounds (can be loose)
    # Max of cos is 1. Sum of A0 over dims + sum of all small_A
    max_y_val = c_val + A0_val * dim + jnp.sum(small_A_val)
    min_y_val = c_val - A0_val * dim - jnp.sum(small_A_val)
    min_y_val = jnp.minimum(min_y_val, max_y_val - 1e-9)

    output_common_params = {
        'type_index': all_possible_names.index(FUNCTION_NAME),
        'optimum_point': optimum_point_val,
        'max_y': max_y_val,
        'min_y': min_y_val,
        'action_dim': dim,
        'bounds': tuple((lower, upper)),
    }

    output_specific_params = jax.tree_util.tree_map(lambda x: x, env_params_instance.sampler_configs['specific'])
    
    spec_entry = output_specific_params[FUNCTION_NAME]
    spec_entry['c_val_sampled'] = c_val
    spec_entry['A0_sampled'] = A0_val
    spec_entry['B0_sampled'] = B0_val
    spec_entry['s0_optimum_sampled'] = s0_val
    spec_entry['small_A_sampled'] = small_A_val
    spec_entry['small_B_sampled'] = small_B_val
    spec_entry['small_shift_sampled'] = small_shift_val
    spec_entry['small_phase_sampled'] = small_phase_val

    return {'common': output_common_params, 'specific': output_specific_params}

def compute_y_func(x: chex.Array, sampler_params: Dict, env_params_instance: 'EnvParams') -> chex.Array:
    """Computes Cosine with Oscillations function value."""
    cos_params = sampler_params['specific'][FUNCTION_NAME]
    common_params = sampler_params['common']
    dim = common_params['action_dim']

    c_val = cos_params['c_val_sampled']
    A0_val = cos_params['A0_sampled']
    B0_val = cos_params['B0_sampled']
    s0_val = cos_params['s0_optimum_sampled']
    small_A_val = cos_params['small_A_sampled']
    small_B_val = cos_params['small_B_sampled']
    small_shift_val = cos_params['small_shift_sampled']
    small_phase_val = cos_params['small_phase_sampled']
    num_oscillations = cos_params['num_oscillations']

    x_eval = x[jnp.newaxis, :] if x.ndim == 1 else x
    
    result = jnp.full(x_eval.shape[0], c_val, dtype=jnp.float64)

    # Main cosine term: A0 * cos(B0_i * (x_i - s0_i)) summed over dimensions
    for i in range(dim):
        result += A0_val * jnp.cos(B0_val[i] * (x_eval[:, i] - s0_val[i]))

    # Small oscillations
    for i in range(dim):
        for k in range(num_oscillations):
            diff = x_eval[:, i] - small_shift_val[i, k]
            result += small_A_val[i, k] * jnp.cos(small_B_val[i, k] * diff + small_phase_val[i, k])
            
    return result.squeeze()
