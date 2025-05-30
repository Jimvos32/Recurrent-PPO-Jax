# kernel_sampler.py
import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Any, List, TYPE_CHECKING, Callable
from jax.experimental import checkify
import gpjax as gpx
# from functools import partial # Not strictly needed here but good for general use

if TYPE_CHECKING:
    # Assuming EnvParams is defined elsewhere and imported, e.g.:
    # from ..jax_disp_samplers import EnvParams
    class EnvParams: pass # Placeholder

GPJAX_GRID_SIZE_DEFAULT = 200

SUPPORTED_GPJAX_KERNELS: Dict[str, Callable[..., gpx.kernels.AbstractKernel]] = {
    "matern52": gpx.kernels.Matern52,
    "matern32": gpx.kernels.Matern32,
    "RBF": gpx.kernels.RBF,
    "polynomial": gpx.kernels.Polynomial,
    "linear": gpx.kernels.Linear,
    "periodic": gpx.kernels.Periodic,
    "white": gpx.kernels.White,
    "arc_cosine": gpx.kernels.ArcCosine,
    "matern12": gpx.kernels.Matern12,
    "exponential": gpx.kernels.PoweredExponential,
    "eigen_comp": gpx.kernels.EigenKernelComputation,
    "rational_quadratic": gpx.kernels.RationalQuadratic,
    "RFF": gpx.kernels.RFF,
    
}

def get_specific_config_template(action_dim: int, run_config: Dict, func_name: str) -> Dict[str, Any]:
    kernel_type_str_for_determining_params = func_name.split('_kernel')[0]
    f_env_config_key = f"{kernel_type_str_for_determining_params}_env"
    func_specific_run_config = run_config.get(f_env_config_key, {})
    grid_size = int(func_specific_run_config.get("grid_size", GPJAX_GRID_SIZE_DEFAULT))
    
    # print("name", f_env_config_key,"split", kernel_type_str_for_determining_params, func_name, "\nfrom", run_config.keys(), "\n")

    template_dict = {
        "grid_size": grid_size,  # Python int, will be made static for JIT
        "x_grid": jnp.full((grid_size,), jnp.nan, dtype=jnp.float64),
        "y_grids": jnp.full((action_dim, grid_size), jnp.nan, dtype=jnp.float64),
        "bounds": tuple(func_specific_run_config.get("bounds", run_config.get("bounds", (-5.0, 5.0)))),
    }
    # print(f"get_specific_config_template: {template_dict}")
  
    
    # if kernel_type_str_for_determining_params == "Polynomial":
    #     template_dict["degree"] = int(func_specific_run_config.get("degree", 1))
    return template_dict

def _create_kernel_instance(kernel_type_str: str, kernel_params: Dict) -> gpx.kernels.AbstractKernel:
    kernel_constructor = SUPPORTED_GPJAX_KERNELS.get(kernel_type_str)
    if not kernel_constructor:
        raise ValueError(f"Kernel type '{kernel_type_str}' not found in SUPPORTED_GPJAX_KERNELS.")
    if kernel_type_str == "Polynomial":
        return kernel_constructor(degree=kernel_params.get("degree", 1))
    else:
        return kernel_constructor()

# MODIFIED: Added grid_size_static argument
def _initialize_inner_gpjax_core(key: chex.PRNGKey,
                                 env_params_instance: 'EnvParams',
                                 action_dim: int,
                                 all_possible_names: Tuple[str, ...],
                                 func_name_static: str, # Static argument
                                 grid_size_static: int # NEW Static argument
                                ) -> Dict:
    """
    Core logic for GPJax kernel-based function initialization.
    grid_size_static is now passed as a static argument.
    """
    kernel_specific_config_from_template = env_params_instance.sampler_configs['specific'][func_name_static]
    

    
    lower, upper = kernel_specific_config_from_template['bounds']
    
    # degree_val = kernel_specific_config_from_template.get('degree') # If needed for polynomial

    derived_kernel_type_str = func_name_static.split('_kernel')[0]
    # This check is more for developer feedback; raising errors in JIT is tricky.
    if derived_kernel_type_str not in SUPPORTED_GPJAX_KERNELS:
         # Consider how to handle this: JAX might replace this with a NaN or fixed value if an error is raised.
         # For now, assume valid func_name_static.
        print(f"Warning: Kernel type {derived_kernel_type_str} not in supported list during JIT trace.")


    derived_kernel_constructor_params = {}
    if derived_kernel_type_str == "Polynomial":
        # 'degree' should be an int in the template, or handled if missing.
        degree = kernel_specific_config_from_template.get('degree')
        if degree is None: # Should be set by get_specific_config_template
            degree = 1
        derived_kernel_constructor_params["degree"] = degree # degree is an int, fine for kernel constructor

    kernel = _create_kernel_instance(derived_kernel_type_str, derived_kernel_constructor_params)
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    # USE THE STATIC grid_size_static ARGUMENT HERE
   
    x_grid_1d = jnp.linspace(lower, upper, num=grid_size_static, dtype=jnp.float64)
    x_grid_gp = x_grid_1d.reshape(-1, 1)
    
    keys = jax.random.split(key, action_dim) # action_dim is already static

    def sample_single_dim(k_single: chex.PRNGKey) -> chex.Array:
        rv = prior(x_grid_gp)
        return rv.sample(key=k_single, sample_shape=(1,)).squeeze(axis=0)

    y_grids_sampled = jax.vmap(sample_single_dim)(keys)

    y_max_per_dim = jnp.max(y_grids_sampled, axis=1)
    y_min_per_dim = jnp.min(y_grids_sampled, axis=1)
    max_y_est = jnp.sum(y_max_per_dim)
    min_y_est = jnp.sum(y_min_per_dim)

    opt_indices_per_dim = jnp.argmax(y_grids_sampled, axis=1)
    optimum_point_est = x_grid_1d[opt_indices_per_dim]
    min_y_final = jnp.minimum(min_y_est, max_y_est - 1e-9)

    output_common_params = {
        'type_index': all_possible_names.index(func_name_static),
        'optimum_point': optimum_point_est,
        'max_y': max_y_est,
        'min_y': min_y_final,
        'action_dim': action_dim,
        'bounds': tuple((lower,upper)),
    }
    

    output_specific_params_all_funcs = jax.tree_util.tree_map(lambda x_leaf: x_leaf, env_params_instance.sampler_configs['specific'])
    
    # Create a mutable copy for the current function's specific config
    # Ensure current_func_specific_config_updated is a dictionary
    current_func_specific_config_original = output_specific_params_all_funcs[func_name_static]
    if not isinstance(current_func_specific_config_original, dict):
        # This case should ideally not happen if templates are dicts.
        # Handle error or ensure it's a dict. For JAX, it expects PyTree leaves.
        # If it's an immutable mapping, convert to dict.
        current_func_specific_config_updated = dict(current_func_specific_config_original)
    else:
        current_func_specific_config_updated = current_func_specific_config_original.copy()

    current_func_specific_config_updated['x_grid'] = x_grid_1d
    current_func_specific_config_updated['y_grids'] = y_grids_sampled
    # grid_size is already in the template, no need to update it here from grid_size_static,
    # as the template value is what's carried in EnvParams.
    # The important part is that grid_size_static was used for array creation.
    output_specific_params_all_funcs[func_name_static] = current_func_specific_config_updated
    
    return {'common': output_common_params, 'specific': output_specific_params_all_funcs}


_checkified_inner_gpjax_init = checkify.checkify(
    _initialize_inner_gpjax_core, errors=checkify.nan_checks
)

# MODIFIED: Added 'grid_size_static' to static_argnames
_jitted_checkified_gpjax_init_base = jax.jit(
    _checkified_inner_gpjax_init,
    static_argnames=('action_dim', 'all_possible_names', 'func_name_static', 'grid_size_static')
)

def initialize_func_template(key: chex.PRNGKey,
                             env_params_instance: 'EnvParams',
                             action_dim: int, 
                             all_possible_names: List[str], 
                             func_name_static: str 
                            ) -> Dict:
    """
    This is the function that will be registered. It extracts grid_size
    and passes it as a static argument to the JITted core logic.
    """
    all_possible_names_tuple = tuple(all_possible_names)

    # Extract grid_size from env_params_instance BEFORE calling the JITted function.
    # This value must be a Python int.
    grid_size_for_static_pass = GPJAX_GRID_SIZE_DEFAULT#env_params_instance.sampler_configs['specific'][func_name_static]['grid_size']
    
    # Ensure it's a Python int if it's somehow a JAX array (e.g. 0-dim) at this point.
    # However, get_specific_config_template should ensure it's an int.
    if not isinstance(grid_size_for_static_pass, int):
        # This path indicates an issue upstream, but as a safeguard:
        try:
            grid_size_for_static_pass = int(grid_size_for_static_pass)
        except TypeError: # Or other errors if it's not convertible
            raise ValueError(
                f"grid_size for {func_name_static} is not a Python int or convertible. "
                f"Got: {grid_size_for_static_pass} (type: {type(grid_size_for_static_pass)})"
            )


    err, out = _jitted_checkified_gpjax_init_base(
        key,
        env_params_instance, 
        action_dim=action_dim, 
        all_possible_names=all_possible_names_tuple, 
        func_name_static=func_name_static,
        grid_size_static=grid_size_for_static_pass # Pass as static argument
    )
    # Handle err if necessary, checkify might populate it with error info
    return out

def compute_y_gpjax_template(x: chex.Array,
                             sampler_params: Dict,
                             env_params_instance: 'EnvParams', 
                             func_name_static: str 
                            ) -> chex.Array:
    kernel_specific_data = sampler_params['specific'][func_name_static]
    x_grid = kernel_specific_data['x_grid']  
    y_grids = kernel_specific_data['y_grids'] 
    x_eval = x if x.ndim == 2 else x[jnp.newaxis, :]

    def interp_1d_single_point(x_scalar_single_dim: chex.Array, y_grid_1d_single_dim: chex.Array) -> chex.Array:
        return jnp.interp(x_scalar_single_dim, x_grid, y_grid_1d_single_dim)

    def process_single_x_row(x_row_all_dims: chex.Array) -> chex.Array: 
        interpolated_values_per_dim = jax.vmap(interp_1d_single_point, in_axes=(0, 0))(x_row_all_dims, y_grids)
        return jnp.sum(interpolated_values_per_dim)

    result_batched = jax.vmap(process_single_x_row, in_axes=0)(x_eval)
    return result_batched.squeeze()
