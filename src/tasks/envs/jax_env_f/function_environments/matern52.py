import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any, List, TYPE_CHECKING

from jax.experimental import checkify
import gpjax as gpx

if TYPE_CHECKING:
    from ..jax_disp_samplers import EnvParams

FUNCTION_NAME = "matern52"
GPJAX_GRID_SIZE_DEFAULT = 200

def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    """Provides the specific config template for Matern52."""
    f_env_config_key = f"{FUNCTION_NAME}_env"
    func_specific_run_config = run_config.get(f_env_config_key, {})
    
    # grid_size is a structural parameter, determined by config, not sampled per instance.
    grid_size = int(func_specific_run_config.get("grid_size", GPJAX_GRID_SIZE_DEFAULT))

    return {
        "grid_size": grid_size, # Store as Python int
        "x_grid": jnp.full((grid_size,), jnp.nan, dtype=jnp.float64),
        "y_grids": jnp.full((action_dim, grid_size), jnp.nan, dtype=jnp.float64),
        "bounds": tuple(func_specific_run_config.get("bounds", run_config.get("bounds", (-5.0, 5.0)))),
    }

# Core logic now takes grid_size directly
def _initialize_inner_matern52_core(key: chex.PRNGKey,
                                    env_params_instance: 'EnvParams',
                                    action_dim: int,
                                    all_possible_names: Tuple[str, ...],
                                    grid_size: int) -> Dict: # Added grid_size
    """
    Core logic for Matern 5/2 initialization.
    """
    # matern_base_config is still used for bounds, but grid_size is now an arg.
    matern_base_config = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]
    lower, upper = matern_base_config['bounds']
    dim = action_dim

    # kernel = gpx.kernels.Matern52()
    kernel = gpx.kernels.Matern52()
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    # Use the passed static grid_size
    x_grid_1d = jnp.linspace(lower, upper, num=grid_size, dtype=jnp.float64)
    x_grid_gp = x_grid_1d.reshape(-1, 1)
    
    keys = jax.random.split(key, dim)

    def sample_single_dim(k_single):
        rv = prior(x_grid_gp)
        return rv.sample(key=k_single, sample_shape=(1,)).squeeze(axis=0)

    y_grids_sampled = jax.vmap(sample_single_dim)(keys)

    y_max_per_dim = jnp.max(y_grids_sampled, axis=1)
    y_min_per_dim = jnp.min(y_grids_sampled, axis=1)
    max_y_est = jnp.sum(y_max_per_dim)
    min_y_est = jnp.sum(y_min_per_dim)

    opt_indices = jnp.argmax(y_grids_sampled, axis=1)
    optimum_point_est = x_grid_1d[opt_indices]

    min_y_final = jnp.minimum(min_y_est, max_y_est - 1e-9)

    output_common_params = {
        'type_index': all_possible_names.index(FUNCTION_NAME),
        'optimum_point': optimum_point_est,
        'max_y': max_y_est,
        'min_y': min_y_final,
        'action_dim': dim,
        'bounds': tuple((lower, upper)),
    }

    output_specific_params = jax.tree_util.tree_map(lambda x_leaf: x_leaf, env_params_instance.sampler_configs['specific'])
    
    output_specific_params[FUNCTION_NAME]['x_grid'] = x_grid_1d
    output_specific_params[FUNCTION_NAME]['y_grids'] = y_grids_sampled
    # 'grid_size' in output_specific_params[FUNCTION_NAME] remains as configured in the template.

    return {'common': output_common_params, 'specific': output_specific_params}

# Wrapper for checkify also takes grid_size
def _checked_inner_matern_init_for_jit(key, env_params_instance, action_dim,
                                       all_possible_names_static: Tuple[str,...],
                                       grid_size: int): # Added grid_size
    return _initialize_inner_matern52_core(key, env_params_instance, action_dim,
                                           all_possible_names_static, grid_size) # Pass grid_size

checkified_matern_init_func = checkify.checkify(
    _checked_inner_matern_init_for_jit, errors=checkify.nan_checks
)

# JITted function now has grid_size as a static argument
_jitted_checkified_matern_init = jax.jit(
    checkified_matern_init_func,
    static_argnames=('action_dim', 'all_possible_names_static', 'grid_size') # Added grid_size
)

def initialize_func(key: chex.PRNGKey,
                    env_params_instance: 'EnvParams',
                    action_dim: int, # This is made static by the Partial in dispatcher
                    all_possible_names: List[str] # This is made static (as a tuple) by the Partial
                   ) -> Dict:
    # Extract grid_size from env_params_instance *before* calling the JITted function.
    # This grid_size will be a Python int, suitable for static argument.
    grid_size_static = env_params_instance.sampler_configs['specific'][FUNCTION_NAME]['grid_size']
    
    # Ensure all_possible_names is a tuple when passed as static_arg
    all_possible_names_tuple = tuple(all_possible_names)

    err, out = _jitted_checkified_matern_init(
        key,
        env_params_instance, # Dynamic argument
        action_dim=action_dim, # Static argument (from Partial)
        all_possible_names_static=all_possible_names_tuple, # Static argument (from Partial, ensured tuple)
        grid_size=grid_size_static # New static argument
    )
    return out

def compute_y_func(x: chex.Array, sampler_params: Dict, env_params_instance: 'EnvParams') -> chex.Array:
    """
    Computes the y value for a given BATCH of x by interpolating the stored GP sample.
    """
    matern_specific_params = sampler_params['specific'][FUNCTION_NAME]
    x_grid = matern_specific_params['x_grid']
    y_grids = matern_specific_params['y_grids']
    # grid_size = matern_specific_params['grid_size'] # Not directly needed for compute_y if x_grid is stored

    x_eval = x[jnp.newaxis, :] if x.ndim == 1 else x

    def interp_1d(x_scalar_single_dim, y_grid_1d_single_dim):
        return jnp.interp(x_scalar_single_dim, x_grid, y_grid_1d_single_dim)

    def process_row(x_row_all_dims):
        interpolated_values_per_dim = jax.vmap(
            interp_1d, in_axes=(0, 0)
        )(x_row_all_dims, y_grids)
        return jnp.sum(interpolated_values_per_dim)

    result_batched = jax.vmap(process_row, in_axes=0)(x_eval)
    
    return result_batched.squeeze()
