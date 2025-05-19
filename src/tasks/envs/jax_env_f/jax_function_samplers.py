import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Tuple, Dict, List, Callable, Any # Keep Any

import flax.struct as struct

# --- Define EnvParams and ActiveFunctionDispatchConfig here ---
# (As in previous correct responses)
@struct.dataclass
class ActiveFunctionDispatchConfig:
    initializers: List[jax.tree_util.Partial]
    computers: List[jax.tree_util.Partial]
    global_to_active_idx_map: chex.Array
    num_active_functions: int = struct.field(pytree_node=False)

@struct.dataclass
class EnvParams:
    # --- Fields that MUST be provided (non-default) ---
    total_samples: int
    max_batches: int # As in your original EnvParams, this was likely an int
    max_steps_in_episode: int
    sampler_configs: Dict[str, Dict[str, Any]] # Full template, non-default
    action_dim: int = struct.field(pytree_node=False) # Now non-default and static for JAX Pytree
    function_type_indices: chex.Array # GLOBAL indices of active functions, non-default
    dispatch_config: ActiveFunctionDispatchConfig # New, non-default
    x_range: Tuple[float, float] = (-5.0, 5.0)
    batches: chex.Array = struct.field(default_factory=lambda: jnp.array([1, 1])) # Or your previous default
    use_random_action_on_reset: bool = True
    use_random_action_on_step: bool = False
    r_scale: float = 10.0
    r_best: float = 0.8
    r_impr: float = 0.1
    r_new_best: float = 0.1
    r_obs: float = 0.0
    r_mse: float = 0.0
    r_suc: float = 3.0
    success_threshold: float = 0.95


# --- Import and Register individual samplers ---
# This section dynamically builds the registries.

# Assuming your new directory is 'function_samplers' sibling to this file
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
from src.tasks.envs.jax_env_f.function_environments import kernel_sampler as gpjax_base_sampler

# ... import others as you create them

GPJAX_FUNCTION_NAMES = ["Matern52", "RBF"] # Example


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
    
    # Add other imported modules here
}

for gpjax_fn_name in GPJAX_FUNCTION_NAMES:
    ALL_FUNCTION_MODULES[gpjax_fn_name] = gpjax_base_sampler
    

# Canonical ordered list of all function names. This order defines global indices.
ALL_POSSIBLE_FUNCTION_NAMES = sorted(ALL_FUNCTION_MODULES.keys())



ALL_FUNCTION_INITIALIZERS_REGISTRY: Dict[str, Callable] = {
    name: module.initialize_func for name, module in ALL_FUNCTION_MODULES.items()
}
ALL_FUNCTION_COMPUTERS_REGISTRY: Dict[str, Callable] = {
    name: module.compute_y_func for name, module in ALL_FUNCTION_MODULES.items()
}
ALL_FUNCTION_CONFIG_TEMPLATE_PROVIDERS: Dict[str, Callable] = {
    name: module.get_specific_config_template for name, module in ALL_FUNCTION_MODULES.items()
}


# --- Helper to get specific config template (uses the new registry) ---
def get_specific_config_template_for_function(func_name: str, action_dim: int, run_config: Dict) -> Dict[str, Any]:
    if func_name in ALL_FUNCTION_CONFIG_TEMPLATE_PROVIDERS:
        return ALL_FUNCTION_CONFIG_TEMPLATE_PROVIDERS[func_name](action_dim, run_config, func_name)
    else:
        print(f"Warning: No specific config template provider for function: {func_name}")
        return {}


# --- _get_jitted_partial_initializer (modified to pass ALL_POSSIBLE_FUNCTION_NAMES) ---
def _get_jitted_partial_initializer(func_name: str, static_action_dim: int) -> jax.tree_util.Partial:
    base_func = ALL_FUNCTION_INITIALIZERS_REGISTRY[func_name]
    # The initialize_func in each module now expects `all_possible_names`
    # Signature: initialize_func(key, env_params_instance, action_dim, all_possible_names)
    # We need to partial out action_dim and all_possible_names
    # The switch branch receives (key, env_params_instance)
    # So, partial_func = partial(base_func, action_dim=static_action_dim, all_possible_names=ALL_POSSIBLE_FUNCTION_NAMES)
    # This assumes ALL_POSSIBLE_FUNCTION_NAMES is a Python list/tuple, static for JIT.
    return jax.tree_util.Partial(
        partial(base_func, action_dim=static_action_dim, all_possible_names=ALL_POSSIBLE_FUNCTION_NAMES)
    )

# _get_partial_computer remains the same as it doesn't need all_possible_names directly for its signature
def _get_partial_computer(func_name: str) -> jax.tree_util.Partial:
    base_func = ALL_FUNCTION_COMPUTERS_REGISTRY[func_name]
    return jax.tree_util.Partial(base_func)


# --- create_env_params (uses the new registries and helpers) ---
def create_env_params(config: Dict) -> EnvParams:
    action_dim_val = config["action_dim"]
    active_function_names = config["function_types"] # List of strings like ["ackley", "poly"]

    active_initializers_list = []
    active_computers_list = []
    active_global_indices_list = []

    for name in active_function_names:
        print(ALL_FUNCTION_MODULES.keys())
        print(f"Registering function: {name}", active_function_names, ALL_FUNCTION_MODULES)
        
        if name not in ALL_FUNCTION_MODULES:
            raise ValueError(f"Function {name} not found in registered modules.")
        active_initializers_list.append(
            _get_jitted_partial_initializer(name, static_action_dim=action_dim_val)
        )
        active_computers_list.append(_get_partial_computer(name))
        active_global_indices_list.append(ALL_POSSIBLE_FUNCTION_NAMES.index(name))

    num_total_possible_functions = len(ALL_POSSIBLE_FUNCTION_NAMES)
    global_to_active_map_list = [-1] * num_total_possible_functions
    for active_idx, global_idx in enumerate(active_global_indices_list):
        global_to_active_map_list[global_idx] = active_idx

    dispatch_config = ActiveFunctionDispatchConfig(
        initializers=active_initializers_list,
        computers=active_computers_list,
        global_to_active_idx_map=jnp.array(global_to_active_map_list, dtype=jnp.int32),
        num_active_functions=len(active_function_names)
    )

    # Build the 'specific' part of the sampler_configs template
    specific_configs_template = {}
    for f_name in ALL_POSSIBLE_FUNCTION_NAMES:
        specific_configs_template[f_name] = get_specific_config_template_for_function(
            f_name, action_dim_val, config
        )
    
    env_params_sampler_configs_template = {
        'common': {
            'type_index': -1,
            'optimum_point': jnp.zeros(action_dim_val, dtype=jnp.float64),
            'max_y': 0.0, 'min_y': 0.0,
            'action_dim': action_dim_val,
            'bounds': tuple(config.get("bounds", (-5.0, 5.0))),
        },
        'specific': specific_configs_template
    }

    # Populate ALL fields of EnvParams
    return EnvParams(
        total_samples=config["total_episode_samples"],
        max_batches=int(config["max_batches"]),
        max_steps_in_episode=config["total_episode_samples"] // int(jnp.min(jnp.array(config["batches"]))),
        sampler_configs=env_params_sampler_configs_template,
        action_dim=action_dim_val,
        function_type_indices=jnp.array(active_global_indices_list, dtype=jnp.int32),
        dispatch_config=dispatch_config,
        x_range=tuple(config.get("bounds", (-5.0, 5.0))),
        batches=jnp.array(config.get("batches", [1,1])), # Provide default for batches
        use_random_action_on_reset=config.get("use_random_action_on_reset", True),
        use_random_action_on_step=config.get("use_random_action_on_step", False),
        r_scale=config.get("r_scale", 10.0),
        r_best=config.get("r_best", 0.8),
        r_impr=config.get("r_impr", 0.1),
        r_new_best=config.get("r_new_best", 0.1),
        r_obs=config.get("r_obs", 0.0),
        r_mse=config.get("r_mse", 0.0),
        r_suc=config.get("r_suc", 3.0),
        success_threshold=config.get("success_threshold", 0.95)
    )

# --- initialize_sampler_dispatch and compute_y_sampler_dispatch ---
# (These remain largely the same as in the previous correct response, using the registries)
def initialize_sampler_dispatch(key: chex.PRNGKey,
                                global_type_index: int,
                                env_params: EnvParams) -> Dict:
    active_idx = env_params.dispatch_config.global_to_active_idx_map[global_type_index]
    # Ensure active_idx is valid before use (e.g. clip or assert if pre-condition is strong)
    safe_active_idx = jnp.clip(active_idx, 0, env_params.dispatch_config.num_active_functions - 1)
    
    # Branches are Partial(initialize_func, action_dim=..., all_possible_names=...)
    # Called as: branch(key, env_params_instance)
    sampler_params_result = jax.lax.switch(
        safe_active_idx,
        env_params.dispatch_config.initializers,
        key,
        env_params
    )
    return sampler_params_result

def compute_y_sampler_dispatch(x: chex.Array,
                               sampler_params_arg: Dict,
                               env_params: EnvParams) -> chex.Array:
    global_type_index = sampler_params_arg['common']['type_index']
    active_idx = env_params.dispatch_config.global_to_active_idx_map[global_type_index]
    safe_active_idx = jnp.clip(active_idx, 0, env_params.dispatch_config.num_active_functions - 1)

    # Branches are Partial(compute_y_func)
    # Called as: branch(x, sampler_params_arg, env_params)
    y_result = jax.lax.switch(
        safe_active_idx,
        env_params.dispatch_config.computers,
        x,
        sampler_params_arg,
        env_params
    )
    return y_result