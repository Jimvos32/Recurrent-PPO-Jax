import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Tuple, Dict, List, Callable, Any 

import flax.struct as struct

# Assuming EnvParams and ActiveFunctionDispatchConfig are defined as in your provided file.
# For brevity, I'll paste them here. Ensure they match your actual definitions.
@struct.dataclass
class ActiveFunctionDispatchConfig:
    initializers: List[jax.tree_util.Partial] # List of callables (Partial objects)
    computers: List[jax.tree_util.Partial]   # List of callables (Partial objects)
    global_to_active_idx_map: chex.Array     # Maps global_type_index to active_function_index
    num_active_functions: int = struct.field(pytree_node=False) # Number of currently active functions

@struct.dataclass
class EnvParams:
    total_samples: int
    max_batches: int
    max_steps_in_episode: int
    sampler_configs: Dict[str, Dict[str, Any]] # PyTree: template, filled by initializers
    action_dim: int = struct.field(pytree_node=False)
    function_type_indices: chex.Array # Global indices of active functions
    dispatch_config: ActiveFunctionDispatchConfig
    x_range: Tuple[float, float] = (-5.0, 5.0)
    batches: chex.Array = struct.field(default_factory=lambda: jnp.array([1, 1], dtype=jnp.int32))
    use_random_action_on_reset: bool = True
    use_random_action_on_step: bool = False
    r_scale: float = 5.0
    r_best: float = 0.1
    r_impr: float = 0.8
    r_new_best: float = 0.1
    r_obs: float = 0.0
    r_mse: float = 0.0
    r_suc: float = 0.0
    r_step_cost: float = -0.1  # Assuming this is a new field for step cost
    success_threshold: float = 0.95


# --- Import and Register individual samplers ---
# Adjust these import paths based on your project structure.
# Example: from .function_environments import ackley as ackley_sampler
# Assuming 'function_environments' is a sibling directory to where this file might be,
# or it's directly under 'src.tasks.envs.jax_env_f' as in your original.

# Non-GPJax Samplers (using your original import paths as a template)
from src.tasks.envs.jax_env_f.function_environments import ackley as ackley_sampler
from src.tasks.envs.jax_env_f.function_environments import poly as poly_sampler
from src.tasks.envs.jax_env_f.function_environments import neg_abs as neg_abs_sampler
from src.tasks.envs.jax_env_f.function_environments import gaussian as gaussian_sampler
from src.tasks.envs.jax_env_f.function_environments import matern52 as matern52_sampler
from src.tasks.envs.jax_env_f.function_environments import cosine as cosine_sampler
from src.tasks.envs.jax_env_f.function_environments import mIchalewicz as michalewicz_sampler
from src.tasks.envs.jax_env_f.function_environments import rosenbrock as rosenbrock_sampler
from src.tasks.envs.jax_env_f.function_environments import eggholder as eggholder_sampler
from src.tasks.envs.jax_env_f.function_environments import branin as branin_sampler
from src.tasks.envs.jax_env_f.function_environments import hartmann as hartmann_sampler
from src.tasks.envs.jax_env_f.function_environments import discrete_peaks as discrete_peaks_sampler
# The new Generic GPJax Kernel Sampler
from src.tasks.envs.jax_env_f.function_environments import kernel_sampler as gpjax_kernel_sampler

# Define names for GPJax kernel-based functions.
# These names will be used as keys in registries and for configuration.
# Ensure they are unique and descriptive.
# The part before "_kernel" should match a key in gpjax_kernel_sampler.SUPPORTED_GPJAX_KERNELS
GPJAX_KERNEL_FUNCTION_NAMES = [
    "matern52_kernel",
    "matern32_kernel",
    "matern12_kernel",
    "RBF_kernel",
    "polynomial_kernel", # For this, ensure 'degree' is in config, e.g. "Polynomial_kernel_env": {"degree": 2}
    "white_kernel",
    "periodic_kernel",
    "linear_kernel",
    "exponential_kernel",
    "rational_quadratic_kernel",
    "arc_cosine_kernel",
    "squared_exponential_kernel",
    "RFF_kernel"
    # You could also have "Polynomial_deg1_kernel", "Polynomial_deg2_kernel" if you prefer distinct names
    # and handle the degree implicitly or via different config blocks.
]

ALL_FUNCTION_MODULES = {
    ackley_sampler.FUNCTION_NAME: ackley_sampler,
    poly_sampler.FUNCTION_NAME: poly_sampler,
    neg_abs_sampler.FUNCTION_NAME: neg_abs_sampler,
    gaussian_sampler.FUNCTION_NAME: gaussian_sampler,
    matern52_sampler.FUNCTION_NAME: matern52_sampler,
    cosine_sampler.FUNCTION_NAME: cosine_sampler,
    michalewicz_sampler.FUNCTION_NAME: michalewicz_sampler,
    rosenbrock_sampler.FUNCTION_NAME: rosenbrock_sampler,
    eggholder_sampler.FUNCTION_NAME: eggholder_sampler,
    branin_sampler.FUNCTION_NAME: branin_sampler,
    hartmann_sampler.FUNCTION_NAME: hartmann_sampler,
    discrete_peaks_sampler.FUNCTION_NAME: discrete_peaks_sampler,
    
    # Add other imported modules here
}


# Add GPJax kernel samplers: each "kernel function name" maps to the same generic module
for gp_func_name in GPJAX_KERNEL_FUNCTION_NAMES:
    ALL_FUNCTION_MODULES[gp_func_name] = gpjax_kernel_sampler


# Canonical ordered list of all function names. This order defines global indices.
ALL_POSSIBLE_FUNCTION_NAMES = sorted(list(ALL_FUNCTION_MODULES.keys()))

# print("All possible function names:", ALL_POSSIBLE_FUNCTION_NAMES)
# --- Registries for initializers, computers, config templates ---
ALL_FUNCTION_INITIALIZERS_REGISTRY: Dict[str, Callable] = {}
ALL_FUNCTION_COMPUTERS_REGISTRY: Dict[str, Callable] = {}
ALL_FUNCTION_CONFIG_TEMPLATE_PROVIDERS: Dict[str, Callable] = {}

# Populate registries for all functions
for name, module in ALL_FUNCTION_MODULES.items():
    if name in GPJAX_KERNEL_FUNCTION_NAMES:
        # GPJax Kernel Samplers
        ALL_FUNCTION_CONFIG_TEMPLATE_PROVIDERS[name] = module.get_specific_config_template
        
        # Initializer: partial out func_name_static for the gpjax_kernel_sampler.initialize_func_template
        # The initialize_func_template handles its own JIT/checkify.
        ALL_FUNCTION_INITIALIZERS_REGISTRY[name] = partial(
            module.initialize_func_template,
            func_name_static=name
        )
        
        # Computer: partial out func_name_static for the gpjax_kernel_sampler.compute_y_gpjax_template
        ALL_FUNCTION_COMPUTERS_REGISTRY[name] = partial(
            module.compute_y_gpjax_template,
            func_name_static=name
        )
    else:
        # Standard (non-GPJax) Samplers
        # Assumes these modules have 'initialize_func', 'compute_y_func', 'get_specific_config_template'
        if hasattr(module, 'get_specific_config_template'):
            ALL_FUNCTION_CONFIG_TEMPLATE_PROVIDERS[name] = module.get_specific_config_template
        else:
            # Provide a default empty config template if one isn't defined
            ALL_FUNCTION_CONFIG_TEMPLATE_PROVIDERS[name] = lambda _ad, _rc, _fn: {}
            print(f"Warning: No 'get_specific_config_template' found for {name}. Using empty default.")

        if hasattr(module, 'initialize_func'):
            ALL_FUNCTION_INITIALIZERS_REGISTRY[name] = module.initialize_func
        else:
            raise AttributeError(f"Module for {name} does not have 'initialize_func'")

        if hasattr(module, 'compute_y_func'):
            ALL_FUNCTION_COMPUTERS_REGISTRY[name] = module.compute_y_func
        else:
            raise AttributeError(f"Module for {name} does not have 'compute_y_func'")


# --- Helper to get specific config template (uses the new registry) ---
def get_specific_config_template_for_function(func_name: str, action_dim: int, run_config: Dict) -> Dict[str, Any]:
    provider = ALL_FUNCTION_CONFIG_TEMPLATE_PROVIDERS.get(func_name)
    if provider:
        # The provider (e.g., gpjax_kernel_sampler.get_specific_config_template)
        # gets func_name, allowing it to customize the template (e.g., set kernel_type).
        return provider(action_dim, run_config, func_name)
    else:
        # This should not be hit if ALL_POSSIBLE_FUNCTION_NAMES are covered by providers.
        print(f"Warning: No specific config template provider for function: {func_name}. Returning empty dict.")
        return {}


# --- _get_jitted_partial_initializer ---
def _get_jitted_partial_initializer(func_name: str, static_action_dim: int) -> jax.tree_util.Partial:
    """
    Returns a jax.tree_util.Partial object that wraps the actual initialization function.
    The wrapped function will be called by jax.lax.switch with (key, env_params_instance).
    This function partials out 'action_dim' and 'all_possible_names'.
    """
    base_func = ALL_FUNCTION_INITIALIZERS_REGISTRY[func_name]
    # For GPJax functions, base_func is already:
    #   partial(gpjax_kernel_sampler.initialize_func_template, func_name_static=func_name)
    #   This initialize_func_template handles its own JIT & checkify, and expects:
    #   (key, env_params, action_dim, all_possible_names) - func_name_static is already baked in.
    # For non-GPJax functions (e.g., ackley_sampler.initialize_func), it expects:
    #   (key, env_params, action_dim, all_possible_names)
    #   These functions might handle their own JIT/checkify (like matern52 did) or be simple.

    # The common signature after this step for the callable inside the Partial should be:
    # (key, env_params_instance)
    # So, we partial out action_dim and all_possible_names.
    
    # Ensure all_possible_names is a tuple for JIT static argument compatibility
    # if the underlying functions (like gpjax_kernel_sampler's template) JIT it.
    all_names_tuple = tuple(ALL_POSSIBLE_FUNCTION_NAMES)

    return jax.tree_util.Partial(
        partial(base_func,
                action_dim=static_action_dim,
                all_possible_names=all_names_tuple)
    )

# --- _get_partial_computer ---
def _get_partial_computer(func_name: str) -> jax.tree_util.Partial:
    """
    Returns a jax.tree_util.Partial object that wraps the actual y-computation function.
    The wrapped function will be called by jax.lax.switch with (x, sampler_params, env_params_instance).
    """
    base_func = ALL_FUNCTION_COMPUTERS_REGISTRY[func_name]
    # For GPJax functions, base_func is already:
    #   partial(gpjax_kernel_sampler.compute_y_gpjax_template, func_name_static=func_name)
    #   This expects (x, sampler_params, env_params_instance) - func_name_static is baked in.
    # For non-GPJax functions (e.g., ackley_sampler.compute_y_func), it expects:
    #   (x, sampler_params, env_params_instance)

    # No further arguments need to be partialized out here for the switch call.
    return jax.tree_util.Partial(base_func)


# --- create_env_params (uses the new registries and helpers) ---
def create_env_params(config: Dict) -> EnvParams:
    action_dim_val = config["action_dim"]
    # List of function name strings, e.g., ["ackley", "Matern52_kernel"]
    active_function_names = config["function_types"]

    active_initializers_list = []
    active_computers_list = []
    active_global_indices_list = []
    for name in active_function_names:
        
        if name not in ALL_FUNCTION_INITIALIZERS_REGISTRY or name not in ALL_FUNCTION_COMPUTERS_REGISTRY:
            raise ValueError(f"Function '{name}' not found in registered initializers/computers. "
                             f"Available: {list(ALL_FUNCTION_INITIALIZERS_REGISTRY.keys())}")
        
        active_initializers_list.append(
            _get_jitted_partial_initializer(name, static_action_dim=action_dim_val)
        )
        active_computers_list.append(_get_partial_computer(name))
        try:
            active_global_indices_list.append(ALL_POSSIBLE_FUNCTION_NAMES.index(name))
        except ValueError:
            raise ValueError(f"Function name '{name}' from config not found in ALL_POSSIBLE_FUNCTION_NAMES. "
                             f"Ensure it's registered correctly (e.g., in GPJAX_KERNEL_FUNCTION_NAMES or standard samplers).")


    num_total_possible_functions = len(ALL_POSSIBLE_FUNCTION_NAMES)
    # Map from global function index to its index within the list of *active* functions
    global_to_active_map_list = [-1] * num_total_possible_functions # -1 for inactive
    for active_idx, global_idx in enumerate(active_global_indices_list):
        global_to_active_map_list[global_idx] = active_idx

    dispatch_config = ActiveFunctionDispatchConfig(
        initializers=active_initializers_list,
        computers=active_computers_list,
        global_to_active_idx_map=jnp.array(global_to_active_map_list, dtype=jnp.int32),
        num_active_functions=len(active_function_names)
    )

    # Build the 'specific' part of the sampler_configs template for ALL possible functions
    specific_configs_template = {}
    for f_name in ALL_POSSIBLE_FUNCTION_NAMES:
        specific_configs_template[f_name] = get_specific_config_template_for_function(
            f_name, action_dim_val, config # Pass f_name here
        )
    
    # This is the template that will be part of EnvParams.
    # The actual values (like sampled grids for GPJax) will be filled in by the initializer functions at runtime.
    env_params_sampler_configs_template = {
        'common': { # Common template structure, actual values filled by initializer
            'type_index': -1, # Placeholder, filled by initializer
            'optimum_point': jnp.zeros(action_dim_val, dtype=jnp.float64), # Placeholder
            'max_y': 5.0, 'min_y': 0.0, # Placeholders
            'action_dim': action_dim_val, # Stored as int
            'bounds': tuple(config.get("bounds", (-5.0, 5.0))), # Default bounds from global config
        },
        'specific': specific_configs_template # Contains templates for all possible functions
    }
    
    env_params_sampler_configs_template['common']['type_index'] = tuple(ALL_POSSIBLE_FUNCTION_NAMES).index(active_function_names[0]) if active_function_names else -1

    # Populate ALL fields of EnvParams
    return EnvParams(
        total_samples=config["total_episode_samples"],
        max_batches=int(config["max_batches"]),
        max_steps_in_episode=config["total_episode_samples"] // int(jnp.min(jnp.array(config.get("batches", [1,1]), dtype=jnp.int32))),
        sampler_configs=env_params_sampler_configs_template,
        action_dim=action_dim_val, # Static field
        function_type_indices=jnp.array(active_global_indices_list, dtype=jnp.int32), # Global indices of active functions
        dispatch_config=dispatch_config,
        x_range=tuple(config.get("bounds", (-5.0, 5.0))),
        batches=jnp.array(config.get("batches", [1,1]), dtype=jnp.int32),
        use_random_action_on_reset=config.get("use_random_action_on_reset", True),
        use_random_action_on_step=config.get("use_random_action_on_step", False),
        r_scale=float(config.get("r_scale", 5.0)),
        r_best=float(config.get("r_best", 0.1)),
        r_impr=float(config.get("r_impr", 0.8)),
        r_new_best=float(config.get("r_new_best", 0.1)),
        r_obs=float(config.get("r_obs", 0.0)),
        r_mse=float(config.get("r_mse", 0.0)),
        r_suc=float(config.get("r_suc", 3.0)),
        r_step_cost=float(config.get("r_step_cost", -0.1)), # Assuming this is a new field
        success_threshold=float(config.get("success_threshold", 0.90))
    )


# --- initialize_sampler_dispatch and compute_y_sampler_dispatch ---
# These should remain largely the same as they operate on the prepared dispatch_config.
def initialize_sampler_dispatch(key: chex.PRNGKey,
                                global_type_index: int, # This is the index in ALL_POSSIBLE_FUNCTION_NAMES
                                env_params: EnvParams) -> Dict:
    """Dispatches to the correct sampler initialization function."""
    active_idx = env_params.dispatch_config.global_to_active_idx_map[global_type_index]
    
    # Ensure active_idx is valid for jax.lax.switch.
    # If global_type_index corresponds to an inactive function, active_idx might be -1.
    # jax.lax.switch requires index to be in [0, num_branches - 1].
    # A simple clip might hide errors; ideally, this path is only called for active functions.
    # If active_idx could be -1, you need a defined behavior (e.g., error or default).
    # Assuming global_type_index is always for an intended active function in the current context.
    safe_active_idx = jnp.clip(active_idx, 0, env_params.dispatch_config.num_active_functions - 1)
    
    # The callables in env_params.dispatch_config.initializers are jax.tree_util.Partial objects.
    # They expect (key, env_params_instance) as arguments.
    
    
    sampler_params_result = jax.lax.switch(
        safe_active_idx, # Must be scalar int
        env_params.dispatch_config.initializers, # List of callables (Partials)
        # Operands to the chosen callable:
        key,
        env_params # This is the env_params_instance
    )
    return sampler_params_result

def compute_y_sampler_dispatch(x: chex.Array,
                               sampler_params_arg: Dict, # Contains common.type_index and specific data
                               env_params: EnvParams) -> chex.Array:
    """Dispatches to the correct y-computation function."""
    global_type_index = sampler_params_arg['common']['type_index']
    active_idx = env_params.dispatch_config.global_to_active_idx_map[global_type_index]
    safe_active_idx = jnp.clip(active_idx, 0, env_params.dispatch_config.num_active_functions - 1)

    # The callables in env_params.dispatch_config.computers are jax.tree_util.Partial objects.
    # They expect (x, sampler_params_arg, env_params_instance) as arguments.
    y_result = jax.lax.switch(
        safe_active_idx, # Must be scalar int
        env_params.dispatch_config.computers, # List of callables (Partials)
        # Operands to the chosen callable:
        x,
        sampler_params_arg,
        env_params # This is the env_params_instance
    )
    return y_result
