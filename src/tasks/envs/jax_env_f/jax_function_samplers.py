import jax
import jax.numpy as jnp
import chex
# Import EnvParams from the jax_env module (adjust path if necessary)
# Assuming jax_env.py is in the same directory or accessible via python path
# from src.tasks.envs.jax_env import EnvParams
from typing import Tuple, Dict, NamedTuple, List, Optional
import functools # Ensure imported
import flax.struct as struct # Or use standard dataclasses
import numpy as np # For NaN checks
# from src.tasks.envs.jax_env.env_params import EnvParams # Adjust import as needed


# Note: The EnvParams class definition is REMOVED from this file.
# It should now be defined ONLY in jax_env.py


import jax
import jax.numpy as jnp
import chex
import flax.struct as struct # Or use standard dataclasses
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Dict, Any

import enum

# --- Option 1: Using enum.Enum with automatic integer values ---
# Enum members get assigned unique integer values starting from 1 by default.

        

@struct.dataclass
class EnvParams:
    """Static environment parameters.
       Derived values like max_steps_in_episode should be calculated
       *before* instantiation and passed in.
    """
    # --- Fields involved in calculation (must be provided at init) ---
    total_samples: int
    max_batches: int

    # --- Derived field (now init=True, value passed during init) ---
    max_steps_in_episode: int # No default, must be provided
    sampler_configs: Dict[str, Dict]
    # --- Other fields ---
    x_range: Tuple[float, float] = (-5.0, 5.0)
    # Provide defaults using default_factory or make them required too
    batches: chex.Array = struct.field(default_factory=lambda: jnp.array([1, 1]))
    action_dim: int = 2 # Example default
    use_random_action_on_reset: bool = False
    use_random_action_on_step: bool = False
    function_type_indices: chex.Array = struct.field(default_factory=lambda: jnp.arange(2))
    r_scale: float = 10.0
    r_best: float = 0.8
    r_impr: float = 0.1
    r_new_best: float = 0.1
    r_obs: float = 0.0
    r_mse: float = 0.0
    r_suc: float = 3.0
    success_threshold: float = 0.95


   # NO __post_init__ needed for this calculation anymore
def create_env_params(config) -> EnvParams:
    """Factory function to create EnvParams."""
    # Calculate max_steps_in_episode based on total_samples and max_batches
    # m_batches = int(jnp.max(jnp.array(config["batches"])))
    min_batches = int(jnp.min(jnp.array(config["batches"])))
    max_steps_in_episode = config["total_episode_samples"] // min_batches
    # print("splitting of batches !!!!!!!!!")
    
    # print("sad", config["batches"], config)
    # for k in config.keys():
    #     print(k)
    
    
    fun_ind = [FuncIndices[fun].value for fun in config["function_types"]]
    

    
    sampler_params = {
        'common': {
            'type_index': fun_ind[0], # Assuming Ackley is index 0
            'optimum_point': 0,
            'max_y': 0,
            'min_y': 0,
            'action_dim': config["action_dim"],
        },
        'specific': {
        }
    }
    
    
    
    for function in ["ackley", "poly"]:
            sampler_params['specific'][function] = function_params_to_dict(function, config["action_dim"])
      
    # print("function types", config["function_types"], fun_ind, sampler_params)

   
    return EnvParams(
        total_samples=config["total_episode_samples"],
        batches=jnp.array(config["batches"]),#this gives an error how can I make this correct
        max_batches=int(config["max_batches"]),
        max_steps_in_episode=max_steps_in_episode,
        sampler_configs=sampler_params,
      
        x_range=config["bounds"],
        action_dim=config["action_dim"],
        use_random_action_on_step=config["random"],
        function_type_indices=jnp.array(fun_ind),
        # ... Add any other default values needed ...
    )
    
    




    
    # ... Add any other init=True fields needed ...

# --- Updated Ackley Initialization with Dummy Arrays ---
def initialize_ackley(key: chex.PRNGKey, params: EnvParams, action_dim: int) -> Dict:
    """
    Initializes Ackley function parameters using config from EnvParams.
    Returns a nested dictionary with dummy arrays for padding.
    """
    # Get Ackley specific config from the main EnvParams
    ackley_config = params.sampler_configs["specific"]['ackley']

    key_a, key_b, key_c, key_opt = jax.random.split(key, 4)
    dim = action_dim # Use concrete integer passed in
    lower, upper = params.x_range # General bounds from params
    opt_factor = ackley_config['optimum_range_factor'] # Specific factor

    center_range = (upper - lower) * opt_factor
    offset = (upper - lower - center_range) / 2.0

    # Sample Ackley parameters using bounds from ackley_config
    a = jax.random.uniform(key_a, shape=(), minval=ackley_config['a_bounds'][0], maxval=ackley_config['a_bounds'][1])
    b = jax.random.uniform(key_b, shape=(), minval=ackley_config['b_bounds'][0], maxval=ackley_config['b_bounds'][1])
    c = jax.random.uniform(key_c, shape=(), minval=ackley_config['c_bounds'][0], maxval=ackley_config['c_bounds'][1])
    optimum_point = jax.random.uniform(key_opt, shape=(dim,), minval=lower + offset, maxval=upper - offset)

    # Calculate max/min for Ackley
    max_y = 0.0
    corner = jnp.full((dim,), upper, dtype=jnp.float32)
    z = corner - optimum_point
    sum_sq = jnp.sum(z**2) / dim
    cos_sum = jnp.sum(jnp.cos(c * z)) / dim
    term1 = -a * jnp.exp(-b * jnp.sqrt(sum_sq))
    term2 = -jnp.exp(cos_sum)
    f_val_min = term1 + term2 + a + jnp.exp(1.0)
    min_y_candidate = -f_val_min
    min_y = jnp.minimum(min_y_candidate, max_y - 1e-6)

    # --- Create the unified nested dictionary ---
    sam_con = params.sampler_configs
    sam_con['common'] = {
        'type_index': 0, # Assuming Ackley is index 0
        'optimum_point': optimum_point,
        'max_y': max_y,
        'min_y': min_y,
        'action_dim': dim,
    }
    
    sam_con['specific']['ackley']['a'] = a
    sam_con['specific']['ackley']['b'] = b
    sam_con['specific']['ackley']['c'] = c 
    # {
    #     'a': a,
    #     'b': b,
    #     'c': c,
    # }
    
    return sam_con


# --- Updated Polynomial Initialization with Dummy Arrays ---
def initialize_poly(key: chex.PRNGKey, params: EnvParams, action_dim: int) -> Dict:
    """
    Initializes Polynomial function parameters using config from EnvParams.
    Returns a nested dictionary with dummy arrays for padding.
    """
    # Get Poly specific config from the main EnvParams
    
    poly_config = params.sampler_configs['specific']['poly']
    
    # print("poly_config", poly_config)

    key_c, key_weights, key_xmax = jax.random.split(key, 3)
    dim = action_dim # Use concrete integer passed in
    lower, upper = params.x_range # General bounds from params
    degree = poly_config['degree'] # Specific degree
    opt_factor = poly_config['optimum_range_factor'] # Specific factor

    # Sample Poly parameters using bounds from poly_config
    poly_c = jax.random.uniform(key_c, shape=(), minval=poly_config['c_bounds'][0], maxval=poly_config['c_bounds'][1])
    poly_weights = jax.random.uniform(key_weights, shape=(dim,), minval=poly_config['weight_bounds'][0], maxval=poly_config['weight_bounds'][1])
    poly_x_max = jax.random.uniform(key_xmax, shape=(dim,), minval=lower * opt_factor, maxval=upper * opt_factor)

    # Calculate max/min for Poly
    optimum_point = poly_x_max
    max_y = poly_c
    x_min_coords = jnp.where(jnp.abs(poly_x_max - lower) > jnp.abs(upper - poly_x_max), lower, upper)
    diff_min = x_min_coords - poly_x_max
    weighted_term_min = poly_weights * (diff_min ** degree)
    min_y_candidate = poly_c - jnp.sum(weighted_term_min)
    min_y = jnp.minimum(min_y_candidate, max_y - 1e-6)
    
    sam_con = params.sampler_configs
    
    
    sam_con['common'] = {
        'type_index': 0, # We dont know must be overruled
        'optimum_point': optimum_point,
        'max_y': max_y,
        'min_y': min_y,
        'action_dim': dim,
    }
    
    sam_con['specific']['poly']['c'] = poly_c
    sam_con['specific']['poly']['weights'] = poly_weights # Real Array
    sam_con['specific']['poly']['x_max'] = poly_x_max     # Real Array
    sam_con['specific']['poly']['degree'] = degree
    
   

    
    return sam_con


# --- initialize_sampler remains the same ---
# It dispatches based on type_index and uses functools.partial
def initialize_sampler(key: chex.PRNGKey, type_index: int, action_dim:int, params: EnvParams) -> Dict:
    """
    Dispatches to the correct sampler initialization based on index.
    Handles static action_dim for lax.switch compatibility.
    Returns a nested dictionary with a unified, padded structure.
    """
    # action_dim = params.action_dim # Concrete Python int

    # Ensure the order here matches the type_index assumption (0: Ackley, 1: Poly)
    base_branches: List[callable] = [
        initialize_ackley,  # Index 0
        initialize_poly,    # Index 1
        # Add other initializers here...
    ]
    # print("pasdg", params)
    # # base_branches = params
    # base_branches = params.sample_switch

    partial_branches = [
        functools.partial(func, action_dim=action_dim) for func in base_branches
    ]

    # num_samplers = len(partial_branches)
    # # Ensure type_index is treated as a JAX type for tracing if necessary
    # type_index_jax = jnp.asarray(type_index)
    # type_index_clipped = jnp.clip(type_index_jax, 0, num_samplers - 1)
    # # Pass the full nested sampler_params dictionary
    # jax.debug.print("Type index: {} {}", type_index, type_index)        
    sampler_params = jax.lax.switch(type_index, partial_branches, key, params)
    sampler_params['common']['type_index'] = type_index 
    return sampler_params


# --- compute_y_ackley remains the same (accessing nested params) ---
def compute_y_ackley(x: chex.Array, sampler_params: Dict, env_params: EnvParams) -> chex.Array:
    common_params = sampler_params['common']
    optimum_point = common_params['optimum_point']
    dim = common_params['action_dim']
    ackley_params = sampler_params['specific']['ackley']
    a = ackley_params['a']
    b = ackley_params['b']
    c = ackley_params['c']

    if x.ndim == 1: x = x[jnp.newaxis, :]
    optimum_point_b = jnp.expand_dims(optimum_point, axis=0)
    z = x - optimum_point_b
    sum_sq_term = -a * jnp.exp(-b * jnp.sqrt(jnp.sum(z**2, axis=-1) / dim))
    cos_term = -jnp.exp(jnp.sum(jnp.cos(c * z), axis=-1) / dim)
    y = sum_sq_term + cos_term + a + jnp.exp(1.0)
    return -y.squeeze()

# --- compute_y_poly remains the same (accessing nested params) ---
def compute_y_poly(x: chex.Array, sampler_params: Dict, env_params: EnvParams) -> chex.Array:
    poly_params = sampler_params['specific']['poly']
    c = poly_params['c']
    weights = poly_params['weights']
    x_max = poly_params['x_max']
    degree = poly_params['degree']
    
    if x.ndim == 1: x = x[jnp.newaxis, :]
    x_max_b = jnp.expand_dims(x_max, axis=0)
    diff = x - x_max_b
   
    weighted_term = weights * (diff ** degree)
    result = c - jnp.sum(weighted_term, axis=-1)
    return result.squeeze()

# --- compute_y_sampler remains the same ---
# It extracts type_index from common params and dispatches
def compute_y_sampler(x: chex.Array, sampler_params: Dict, env_params: EnvParams) -> chex.Array:
    """
    Dispatches to the correct compute_y function based on 'type_index'
    stored within the sampler_params['common'] dictionary.
    """
    # Extract type index from the common part
    type_index = sampler_params['common']['type_index']
    # type_index = env_params.function_type_indices

    # Ensure the order matches the type_index assumption (0: Ackley, 1: Poly)
    branches: List[callable] = [
        compute_y_ackley,  # Index 0
        compute_y_poly,    # Index 1
        # Add other compute functions here...
    ]
    
    print("branches", x.shape)
    
    # branches = env_params.compute_switch

    num_samplers = len(branches)
    # Ensure type_index is treated as a JAX type for tracing if necessary
    type_index_jax = jnp.asarray(type_index)
    type_index_clipped = jnp.clip(type_index_jax, 0, num_samplers - 1)
    

    # Pass the full nested sampler_params dictionary
    y = jax.lax.switch(type_index_clipped, branches, x, sampler_params, env_params)

    return y



class FuncIndices(enum.Enum):
    ackley = enum.auto()    # Gets assigned 1
    poly = enum.auto()  # Gets assigned 2
    cosine = enum.auto()   # Gets assigned 3
    branin = enum.auto() # Gets assigned 4
    
class FunctionToSampler(enum.Enum):
    ackley = initialize_ackley
    poly = initialize_poly
    
    
func_to_sampler = {
    "ackley": initialize_ackley,
    "poly": initialize_poly,
}
    
func_to_compute = {
    "ackley": compute_y_ackley,
    "poly": compute_y_poly,
}
    
    
def function_params_to_dict(function: str, dim: int) -> Dict[str, Any]:
    if function == "ackley":
        return {
                "a": 0.2,
                "b": 0.2,
                "c": 0.2,
                'optimum_range_factor': 0.9,
                'a_bounds': (15.0, 20.0),
                'b_bounds': (0.1, 0.2),
                'c_bounds': (2 * np.pi, 2 * np.pi),
            }
           
    if function == "poly":
        return {
                'c': jnp.nan,
                'weights': jnp.full((dim,), jnp.nan, dtype=jnp.float32), # Dummy Array
                'x_max': jnp.full((dim,), jnp.nan, dtype=jnp.float32),   # Dummy Array
                'degree': 0,
                'optimum_range_factor': 0.9,
                'c_bounds': (0.0, 1.0),
                'weight_bounds': (0.0, 1.0),
                
            }
      