# import jax
# import jax.numpy as jnp
# import chex
# from typing import Dict, Tuple, Any, List, TYPE_CHECKING

# # No need to import 'partial' from functools here unless used for other purposes in this file.
# # jax.jit itself can serve the role of partial for static_argnames.

# if TYPE_CHECKING:
#     from ..jax_disp_samplers import EnvParams # Adjust path as per your final structure

# FUNCTION_NAME = "discrete_peaks"
# FIXED_NUM_PEAK_LOCATIONS = 5 # Default number of peak locations per dimension

# def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
#     """Provides the specific config template for the Discrete Peak function."""
#     f_env_config_key = f"{FUNCTION_NAME}_env"
#     func_specific_run_config = run_config.get(f_env_config_key, {})
#     default_bounds = run_config.get("bounds", (-1.0, 1.0))

#     # Ensure num_peak_locations_per_dim is stored as a Python int

#     return {
#         'num_peak_locations_per_dim': FIXED_NUM_PEAK_LOCATIONS, # Stored as Python int
#         'peak_amplitude': float(func_specific_run_config.get('peak_amplitude', 1.0)),
#         'peak_sharpness_factor': float(func_specific_run_config.get('peak_sharpness_factor', 0.1)),
#         'baseline_value': float(func_specific_run_config.get('baseline_value', 0.0)),
#         'bounds': tuple(func_specific_run_config.get("bounds", default_bounds)),
#     }

# def _calculate_discrete_locations_for_dim(min_val: float,
#                                           max_val: float,
#                                           num_locs: int) -> jnp.ndarray:
#     """
#     Calculates N evenly spaced locations for a single dimension, not at the edges.
#     num_locs is a Python int here.
#     """
#     num_locs = FIXED_NUM_PEAK_LOCATIONS
    
#     step_size = (max_val - min_val) / (num_locs + 1)
#     locations = [min_val + (i + 1) * step_size for i in range(num_locs)]
#     return jnp.array(locations, dtype=jnp.float64)

# def _initialize_inner_discrete_peak_core(key: chex.PRNGKey,
#                                          env_params_instance: 'EnvParams',
#                                          action_dim: int, # Static argument for JIT
#                                          all_possible_names_tuple: Tuple[str, ...], # Static for JIT
#                                          num_locs_per_dim_static: int # Static for JIT
#                                          ) -> Dict:
#     """Core logic for Discrete Peak initialization."""
#     num_locs_per_dim_static = FIXED_NUM_PEAK_LOCATIONS
    
    
#     # Config is accessed for other dynamic parameters (bounds, amplitude, etc.)
#     config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]
#     dim = action_dim # Will be a static Python int

#     bounds = config['bounds']
#     min_val, max_val = bounds[0], bounds[1]

#     # num_locs_per_dim_static is a Python int, so _calculate_discrete_locations_for_dim can use range()
#     possible_locs_1d_list = [
#         _calculate_discrete_locations_for_dim(min_val, max_val, num_locs_per_dim_static)
#         for _ in range(dim) # dim is static, so this Python loop unrolls
#     ]

#     grids = jnp.meshgrid(*possible_locs_1d_list, indexing='ij')
#     all_possible_nd_optima = jnp.stack(grids, axis=-1).reshape(-1, dim)

#     key_select, _ = jax.random.split(key) # Not using second key for now
#     num_total_possible_optima = all_possible_nd_optima.shape[0]
    
#     selected_idx = jax.random.randint(key_select, shape=(), minval=0, maxval=jnp.maximum(1, num_total_possible_optima))
#     current_optimum_point = all_possible_nd_optima[selected_idx]

#     peak_amplitude = jnp.asarray(config['peak_amplitude'], dtype=jnp.float64)
#     baseline_value = jnp.asarray(config['baseline_value'], dtype=jnp.float64)
    
#     _step = (jnp.asarray(max_val, dtype=jnp.float64) - jnp.asarray(min_val, dtype=jnp.float64)) / \
#             jnp.maximum(1.0, (jnp.asarray(num_locs_per_dim_static, dtype=jnp.float64) + 1.0))
            
#     peak_sharpness = jnp.asarray(_step * config['peak_sharpness_factor'], dtype=jnp.float64)
#     peak_sharpness = jnp.maximum(peak_sharpness, 1e-6)

#     max_y_val = baseline_value + peak_amplitude
#     min_y_val = baseline_value

#     # all_possible_names_tuple is already a tuple
#     type_idx = all_possible_names_tuple.index(FUNCTION_NAME)

#     output_common_params = {
#         'type_index': type_idx,
#         'optimum_point': current_optimum_point,
#         'max_y': max_y_val,
#         'min_y': min_y_val,
#         'action_dim': dim,
#         'bounds': tuple((min_val, max_val)),
#     }

#     # Create a new dictionary for specific params to avoid modifying the template in env_params_instance directly
#     output_specific_params = jax.tree_util.tree_map(lambda x: x, env_params_instance.sampler_configs['specific'])
    
#     # Update only this function's specific sampled parameters
#     # Note: 'num_peak_locations_per_dim' from the template is not re-assigned here if it's purely static for init logic
#     output_specific_params[FUNCTION_NAME].update({
#         'current_optimum_point': current_optimum_point,
#         'peak_amplitude': peak_amplitude,
#         'peak_sharpness': peak_sharpness,
#         'baseline_value': baseline_value,
#     })
#     # The original 'num_peak_locations_per_dim' (the Python int) remains in output_specific_params[FUNCTION_NAME]
#     # if it was part of the template copied by tree_map, which is good for consistency.

#     return {'common': output_common_params, 'specific': output_specific_params}

# # JIT-compile the core logic with specified static arguments
# _jitted_inner_discrete_peak_init = jax.jit(
#     _initialize_inner_discrete_peak_core,
#     static_argnames=('action_dim', 'all_possible_names_tuple', 'num_locs_per_dim_static')
# )

# def initialize_func(key: chex.PRNGKey,
#                     env_params_instance: 'EnvParams',
#                     action_dim: int, # Made static by the Partial in dispatcher
#                     all_possible_names: List[str] # Made static (as tuple) by dispatcher's Partial
#                    ) -> Dict:
#     """
#     Main initialization function called by the dispatcher.
#     It extracts static params and calls the JITted core logic.
#     """
#     # Extract num_locs_per_dim as a Python int from env_params_instance.
#     # This relies on it being stored as an int and accessible as such here.
#     num_locs_static = FIXED_NUM_PEAK_LOCATIONS
    
#     # Basic check to ensure it's a Python int as expected for a static argument
#     # if not isinstance(num_locs_static, int):
#     #     raise TypeError(
#     #         f"Expected num_peak_locations_per_dim to be an int for static argument, "
#     #         f"but got {type(num_locs_static)}. Value: {num_locs_static}"
#     #     )

#     # Ensure all_possible_names is a tuple for the static argument of the JITted function
#     all_possible_names_tuple = tuple(all_possible_names)

#     # Call the JITted inner function
#     return _jitted_inner_discrete_peak_init(
#         key,
#         env_params_instance, # Dynamic argument
#         action_dim=action_dim, # Static argument
#         all_possible_names_tuple=all_possible_names_tuple, # Static argument
#         num_locs_per_dim_static=num_locs_static # Static argument
#     )

# def compute_y_func(x: chex.Array,
#                    sampler_params: Dict,
#                    env_params_instance: 'EnvParams') -> chex.Array:
#     """Computes the Discrete Peak function value."""
#     # This function remains unchanged.
#     specific_params = sampler_params['specific'][FUNCTION_NAME]

#     optimum_point = specific_params['current_optimum_point']
#     amplitude = specific_params['peak_amplitude']
#     sharpness = specific_params['peak_sharpness']
#     baseline = specific_params['baseline_value']

#     x_eval = x[jnp.newaxis, :] if x.ndim == 1 else x

#     diff = x_eval - optimum_point
#     dist_sq_scaled = jnp.sum((diff / sharpness)**2, axis=-1)
#     peak_value = amplitude * jnp.exp(-0.5 * dist_sq_scaled)
#     result = baseline + peak_value

#     return result.squeeze()


import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any, List, TYPE_CHECKING
from functools import partial # Required for the JIT decorator

if TYPE_CHECKING:
    # Adjust this import path based on your project structure
    from ..jax_disp_samplers import EnvParams

FUNCTION_NAME = "discrete_peaks"
FIXED_NUM_PEAK_LOCATIONS = 5 # Default number of peak locations per dimension

def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    """Provides the specific config template for the Discrete Peak function."""
    f_env_config_key = f"{FUNCTION_NAME}_env"
    func_specific_run_config = run_config.get(f_env_config_key, {})
    default_bounds = run_config.get("bounds", (-1.0, 1.0))

    # num_peak_locations_per_dim is a structural parameter, determined by config.
    num_locs = int(func_specific_run_config.get('num_peak_locations_per_dim', 5))
    
    

    return {
        'num_peak_locations_per_dim': num_locs, # Stored as Python int
        'peak_amplitude': float(func_specific_run_config.get('peak_amplitude', 1.0)),
        'peak_sharpness_factor': float(func_specific_run_config.get('peak_sharpness_factor', 0.1)),
        'baseline_value': float(func_specific_run_config.get('baseline_value', 0.0)),
        'bounds': tuple(func_specific_run_config.get("bounds", default_bounds)),
        # New parameters for global slope and peak offset
        'global_slope_bounds': tuple(func_specific_run_config.get('global_slope_bounds', (0.02, 0.05))),
        'peak_offset_factor': float(func_specific_run_config.get('peak_offset_factor', 0.1)), # Fraction of inter-discrete-location spacing
    }

def _calculate_discrete_locations_for_dim(min_val: float,
                                          max_val: float,
                                          num_locs: int) -> jnp.ndarray:
    """
    Calculates N evenly spaced locations for a single dimension, not at the edges.
    num_locs is a Python int here.
    """
    if num_locs <= 0:
        return jnp.array([(min_val + max_val) / 2.0], dtype=jnp.float64)
    
    step_size = (max_val - min_val) / (num_locs + 1)
    locations = [min_val + (i + 1) * step_size for i in range(num_locs)]
    return jnp.array(locations, dtype=jnp.float64)

# This is the JIT-compiled core logic.
@partial(jax.jit, static_argnames=('action_dim', 'all_possible_names_tuple', 'num_locs_per_dim_static'))
def _initialize_inner_discrete_peak_core(key: chex.PRNGKey,
                                         env_params_instance: 'EnvParams',
                                         action_dim: int, 
                                         all_possible_names_tuple: Tuple[str, ...], 
                                         num_locs_per_dim_static: int
                                         ) -> Dict:
    """Core logic for Discrete Peak initialization with global slope and peak offset."""
    
    config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]
    dim = action_dim 

    bounds = config['bounds']
    min_val, max_val = bounds[0], bounds[1]

    key_select, key_offset, key_slope = jax.random.split(key, 3)

    # 1. Determine discrete center for the peak
    possible_locs_1d_list = [
        _calculate_discrete_locations_for_dim(min_val, max_val, num_locs_per_dim_static)
        for _ in range(dim) 
    ]
    grids = jnp.meshgrid(*possible_locs_1d_list, indexing='ij')
    all_possible_discrete_centers = jnp.stack(grids, axis=-1).reshape(-1, dim)
    
    num_total_possible_centers = all_possible_discrete_centers.shape[0]
    selected_center_idx = jax.random.randint(key_select, shape=(), minval=0, maxval=jnp.maximum(1, num_total_possible_centers))
    selected_discrete_center = all_possible_discrete_centers[selected_center_idx]

    # 2. Add random offset to the selected discrete center
    # _step calculation is for the characteristic spacing
    _step = (jnp.asarray(max_val, dtype=jnp.float64) - jnp.asarray(min_val, dtype=jnp.float64)) / \
            jnp.maximum(1.0, (jnp.asarray(num_locs_per_dim_static, dtype=jnp.float64) + 1.0))
    
    offset_max_magnitude_per_dim = _step * config['peak_offset_factor']
    random_offset = jax.random.uniform(key_offset, 
                                       shape=(dim,), 
                                       minval=-offset_max_magnitude_per_dim, 
                                       maxval=offset_max_magnitude_per_dim,
                                       dtype=jnp.float64)
    
    final_optimum_point = selected_discrete_center + random_offset
    # Clip the final optimum point to stay within bounds
    final_optimum_point = jnp.clip(final_optimum_point, min_val, max_val)


    # 3. Sample global slope
    slope_min, slope_max = config['global_slope_bounds']
    global_slope_val = jax.random.uniform(key_slope, shape=(), minval=slope_min, maxval=slope_max, dtype=jnp.float64)

    # 4. Other parameters
    peak_amplitude = jnp.asarray(config['peak_amplitude'], dtype=jnp.float64)
    baseline_value = jnp.asarray(config['baseline_value'], dtype=jnp.float64)
    peak_sharpness = jnp.asarray(_step * config['peak_sharpness_factor'], dtype=jnp.float64)
    peak_sharpness = jnp.maximum(peak_sharpness, 1e-6) # Ensure positive sharpness

    # 5. Calculate max_y and min_y considering the global slope
    # Max y is at the peak (Gaussian peak value + baseline, slope term is 0 at optimum)
    max_y_val = baseline_value + peak_amplitude 

    # Min y: at the corner furthest from final_optimum_point
    # The Gaussian peak contribution is assumed negligible at corners.
    # y_corner ~ baseline_value - global_slope * distance_to_corner
    corners = jnp.array(jnp.meshgrid(*([jnp.array([min_val, max_val])] * dim)), dtype=jnp.float64).T.reshape(-1, dim)
    distances_to_corners = jnp.linalg.norm(corners - final_optimum_point, axis=1)
    max_dist_to_corner = jnp.max(distances_to_corners)
    min_y_val = baseline_value - global_slope_val * max_dist_to_corner
    min_y_val = jnp.minimum(min_y_val, max_y_val - 1e-6) # Ensure min_y < max_y

    type_idx = all_possible_names_tuple.index(FUNCTION_NAME)

    output_common_params = {
        'type_index': type_idx,
        'optimum_point': final_optimum_point, # This is the true optimum
        'max_y': max_y_val,
        'min_y': min_y_val,
        'action_dim': dim,
        'bounds': tuple((min_val, max_val)),
    }
    
    output_specific_params = jax.tree_util.tree_map(lambda x: x, env_params_instance.sampler_configs['specific'])
    output_specific_params[FUNCTION_NAME].update({
        'current_optimum_point': final_optimum_point, # Renamed for clarity from previous versions
        'peak_amplitude': peak_amplitude,
        'peak_sharpness': peak_sharpness,
        'baseline_value': baseline_value,
        'global_slope': global_slope_val, # Store sampled global slope
        # num_peak_locations_per_dim is already in config, and used as static.
        # peak_offset_factor and global_slope_bounds are also in config.
    })

    return {'common': output_common_params, 'specific': output_specific_params}


def initialize_func(key: chex.PRNGKey,
                    env_params_instance: 'EnvParams',
                    action_dim: int, 
                    all_possible_names: List[str] 
                   ) -> Dict:
    """
    Main initialization function called by the dispatcher.
    It extracts static params and calls the JITted core logic.
    """
    # num_locs_static is fetched from the config template within env_params_instance.
    # This relies on get_specific_config_template storing it as a Python int.
    num_locs_static = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]['num_peak_locations_per_dim']
    
    num_locs_static = FIXED_NUM_PEAK_LOCATIONS
    
    # if not isinstance(num_locs_static, int):
    #     # This check is good practice, though the dispatcher setup should ensure it.
    #     raise TypeError(
    #         f"Expected num_peak_locations_per_dim to be an int for static argument, "
    #         f"but got {type(num_locs_static)}. Value: {num_locs_static}"
    #     )

    all_possible_names_tuple = tuple(all_possible_names)

    return _initialize_inner_discrete_peak_core(
        key,
        env_params_instance, 
        action_dim=action_dim, 
        all_possible_names_tuple=all_possible_names_tuple, 
        num_locs_per_dim_static=num_locs_static 
    )
    
    
# def initialize_func(key: chex.PRNGKey,
#                     env_params_instance: 'EnvParams',
#                     action_dim: int, # Made static by the Partial in dispatcher
#                     all_possible_names: List[str] # Made static (as tuple) by dispatcher's Partial
#                    ) -> Dict:
#     """
#     Main initialization function called by the dispatcher.
#     It extracts static params and calls the JITted core logic.
#     """
#     # Extract num_locs_per_dim as a Python int from env_params_instance.
#     # This relies on it being stored as an int and accessible as such here.
#     num_locs_static = FIXED_NUM_PEAK_LOCATIONS
    
#     # Basic check to ensure it's a Python int as expected for a static argument
#     # if not isinstance(num_locs_static, int):
#     #     raise TypeError(
#     #         f"Expected num_peak_locations_per_dim to be an int for static argument, "
#     #         f"but got {type(num_locs_static)}. Value: {num_locs_static}"
#     #     )

#     # Ensure all_possible_names is a tuple for the static argument of the JITted function
#     all_possible_names_tuple = tuple(all_possible_names)

#     # Call the JITted inner function
#     return _jitted_inner_discrete_peak_init(
#         key,
#         env_params_instance, # Dynamic argument
#         action_dim=action_dim, # Static argument
#         all_possible_names_tuple=all_possible_names_tuple, # Static argument
#         num_locs_per_dim_static=num_locs_static # Static argument
#     )


def compute_y_func(x: chex.Array,
                   sampler_params: Dict,
                   env_params_instance: 'EnvParams') -> chex.Array:
    """Computes the Discrete Peak function value, including global slope."""
    specific_params = sampler_params['specific'][FUNCTION_NAME]

    optimum_point = specific_params['current_optimum_point']
    amplitude = specific_params['peak_amplitude']
    sharpness = specific_params['peak_sharpness']
    baseline = specific_params['baseline_value']
    global_slope = specific_params['global_slope']

    x_eval = x[jnp.newaxis, :] if x.ndim == 1 else x # Ensure (batch, dim)

    # Gaussian peak component
    diff_for_gaussian = x_eval - optimum_point 
    dist_sq_scaled = jnp.sum((diff_for_gaussian / sharpness)**2, axis=-1) # Sum over dimensions -> (batch,)
    gaussian_component = baseline + amplitude * jnp.exp(-0.5 * dist_sq_scaled)

    # Global slope component: -slope * distance_to_optimum
    # distance_to_optimum = jnp.linalg.norm(diff_for_gaussian, axis=-1) # (batch,)
    # Using a robust norm calculation:
    distance_to_optimum = jnp.sqrt(jnp.sum(diff_for_gaussian**2, axis=-1) + 1e-9) # Add epsilon for stability if diff can be zero

    slope_component = global_slope * distance_to_optimum
    
    result = gaussian_component - slope_component # Slope term makes it go up towards peak

    return result.squeeze()
