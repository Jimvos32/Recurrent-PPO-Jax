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
    use_random_action_on_reset: bool = True
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
    
    
    
    for function in ["ackley", "poly", "neg_abs", "gaussian"]:
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
    # jax.debug.print("init_ack {} {}", params.sampler_configs['common']['type_index'], sam_con['common']['min_y'])
    
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
    poly_steep = jax.random.uniform(key_xmax, shape=(), minval=poly_config['steep_min'], maxval=poly_config['steep_max'] * 2.0)

    # Calculate max/min for Poly
    optimum_point = poly_x_max
    max_y = poly_c
    x_min_coords = jnp.where(jnp.abs(poly_x_max - lower) > jnp.abs(upper - poly_x_max), lower, upper)
    diff_min = x_min_coords - poly_x_max
    weighted_term_min = poly_weights * (diff_min ** degree)
    min_y_candidate = poly_c - poly_steep * jnp.sum(weighted_term_min)
    min_y = jnp.minimum(min_y_candidate, max_y - 1e-6)
    
    sam_con = params.sampler_configs
    
    
    sam_con['common'] = {
        'type_index': 1, # We dont know must be overruled
        'optimum_point': optimum_point,
        'max_y': max_y,
        'min_y': min_y,
        'action_dim': dim,
    }
    
    sam_con['specific']['poly']['c'] = poly_c
    sam_con['specific']['poly']['weights'] = poly_weights # Real Array
    sam_con['specific']['poly']['x_max'] = poly_x_max     # Real Array
    sam_con['specific']['poly']['degree'] = degree
    sam_con['specific']['poly']['steepness_factor'] = poly_steep # Real Array
    
   
    # jax.debug.print("init_pol {} {}", params.sampler_configs['common']['type_index'], sam_con['common']['min_y'])
    
    return sam_con


# Add these functions to jax_function_samplers.py

# --- Negative Absolute Value Function ---

def initialize_neg_abs(key: chex.PRNGKey, params: EnvParams, action_dim: int) -> Dict:
    """
    Initializes parameters for the Negative Absolute Value function f(x) = -sum(|x_i|).
    This function has no specific tunable parameters but calculates bounds.
    Returns a nested dictionary matching the expected structure.
    """
    
    # No specific parameters to sample for this function.
    dim = action_dim
    lower, upper = params.x_range

    # Optimum point is always at the origin for this function.
    optimum_point = jnp.zeros((dim,), dtype=jnp.float32)

    # Maximum value is always 0 at the optimum point.
    max_y = 0.0

    # Minimum value occurs at the corners furthest from the origin.
    # Calculate -sum(|corner_coord|) for one such corner.
    # Assumes symmetric bounds [-L, L] or [0, L] or [-L, 0].
    # Find the corner coordinate with the largest absolute value in each dimension.
    corner_coord_abs_max = jnp.maximum(jnp.abs(lower), jnp.abs(upper))
    
    
    min_y_candidate = -jnp.sum(jnp.full((dim,), corner_coord_abs_max))

    # Ensure min_y is slightly less than max_y if bounds are very small
    min_y = jnp.minimum(min_y_candidate, max_y - 1e-6)

    # --- Create the unified nested dictionary ---
    # Get the base structure, potentially modifying common fields
    # It's safer to create a new dict or carefully update a copy
    # Let's assume we need to populate the structure like the others.
    # Get the default structure for padding/consistency if needed from params.sampler_configs
    sam_con = jax.tree_map(lambda x: x, params.sampler_configs) # Create a copy

    sam_con['common'] = {
        'type_index': 3, # Placeholder, will be set by initialize_sampler
        'optimum_point': optimum_point,
        'max_y': max_y,
        'min_y': min_y,
        'action_dim': dim,
    }
    
    
    
    # print("sam_con", min_y, max_y, optimum_point, dim, "gsdfg")
    # Add an empty entry for 'neg_abs' under specific for structural consistency if needed
    # Or ensure the padding mechanism handles missing keys gracefully.
    # If function_params_to_dict creates padding, we might not need this.
    # Let's assume the structure needs the key:
    if 'neg_abs' not in sam_con['specific']:
         sam_con['specific']['neg_abs'] = {} # Empty dict, no specific params
         
    # jax.debug.print("corner_coord_abs_max {} {} {}\ncalc min {} max {}\n conning to resutn{}\n", corner_coord_abs_max, lower, upper, min_y, max_y, sam_con)
    # jax.debug.print("init_neg {} {}", params.sampler_configs['common']['type_index'], sam_con['common']['min_y'])

    return sam_con

def compute_y_neg_abs(x: chex.Array, sampler_params: Dict, env_params: EnvParams) -> chex.Array:
    """
    Computes the Negative Absolute Value function f(x) = -sum(|x_i|).
    """
    
    
    # No specific parameters needed from sampler_params for this function.

    # Ensure x has a batch dimension if needed
    if x.ndim == 1: x = x[jnp.newaxis, :] # Add batch dim if input is single vector

    # Calculate -sum(|x_i|) along the last axis (dimension axis)
    result = -jnp.sum(jnp.abs(x), axis=-1)

    # Optional debug print
    # common_params = sampler_params['common']
    # jax.debug.print("COMPUTE_NEG_ABS: Input x[0]={x_val}, Result={res_val}, Min={min_v}, Max={max_v}",
    #                 x_val=x[0,0], res_val=result.squeeze(), min_v=common_params['min_y'], max_v=common_params['max_y'])
    
    
    # jax.debug.print("x {} result {} min {} max {}\n",x, result, sampler_params['common']['min_y'], sampler_params['common']['max_y'])

    return result.squeeze() # Remove batch dim if input was single vector


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
        initialize_neg_abs,
        initialize_gaussian,
        # Add other initializers here...
    ]
    # print("pasdg", params)
    # # base_branches = params
    # base_branches = params.sample_switch
    
    

    partial_branches = [
        functools.partial(func, action_dim=action_dim) for func in base_branches
    ]
    type_index = type_index - 1
    num_samplers = len(partial_branches)
    # Ensure type_index is treated as a JAX type for tracing if necessary
    type_index_jax = jnp.asarray(type_index)
    type_index_clipped = jnp.clip(type_index_jax, 0, num_samplers - 1)

 
    sampler_params = jax.lax.switch(type_index_clipped, partial_branches, key, params)
    sampler_params['common']['type_index'] = type_index 
    
    # jax.debug.print("init_general {}  {} {} {}", params.sampler_configs['common']['type_index'], type_index, type_index_clipped, sampler_params['common']['min_y'])
    
    # jax.debug.print("sampler_params {} afeter init", sampler_params)
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
    # x = jnp.full_like(x, 0.5, dtype=jnp.float32)  # Dummy array for padding
    
    # print("compute_y_poly", x.shape, x.dtype, sampler_params['common']['type_index'])
    poly_params = sampler_params['specific']['poly']
    c = poly_params['c']
    weights = poly_params['weights']
    x_max = poly_params['x_max']
    degree = poly_params['degree']
    steep = poly_params['steepness_factor']
    
    if x.ndim == 1: x = x[jnp.newaxis, :]
    x_max_b = jnp.expand_dims(x_max, axis=0)
    diff = x - x_max_b
   
    weighted_term = weights * (diff ** degree)
    result = c - steep * jnp.sum(weighted_term, axis=-1)
    
    # jax.debug.print("Poly x {} c {} weights {} x_max {} degree {} x_max_b {} diff {} weighted_term {} result {} min{}]",
                    # x, c, weights, x_max, degree, x_max_b, diff, weighted_term, result, sampler_params['common']['min_y'])
    
    # jax.debug.print("\ncompute_y_poly x {} result {} x_max {}\nmin {} max {}\n",
    #                 x, result, x_max, sampler_params['common']['min_y'], sampler_params['common']['max_y'])
    return result.squeeze()

# --- compute_y_sampler remains the same ---
# It extracts type_index from common params and dispatches



def compute_y_gaussian(x: chex.Array, sampler_params: Dict, env_params: EnvParams) -> chex.Array:
    """
    Computes the y value for a given x using the sampled Gaussian parameters.
    """
    # Extract Gaussian parameters
    gauss_params = sampler_params['specific']['gaussian']
    center = gauss_params['center']
    width = gauss_params['width']
    amplitude = gauss_params['amplitude']
    baseline = gauss_params['baseline']

    # Ensure x has a batch dimension for broadcasting (if needed)
    if x.ndim == 1:
        x = x[jnp.newaxis, :] # Add batch dimension

    # Center x relative to the peak location
    # Broadcasting works: x(batch, dim), center(dim) -> diff(batch, dim)
    diff = x - center

    # Calculate the squared difference scaled by width
    # Broadcasting works: diff(batch, dim), width(dim) -> scaled_diff_sq(batch, dim)
    scaled_diff_sq = (diff / width) ** 2

    # Calculate the exponent (sum over dimensions)
    # Sum over the last axis (dimension axis) -> exponent(batch,)
    exponent = -0.5 * jnp.sum(scaled_diff_sq, axis=-1)

    # Calculate the final result
    # Broadcasting works: baseline(scalar), amplitude(scalar), exp(exponent)(batch,) -> result(batch,)
    result = baseline + amplitude * jnp.exp(exponent)

    # jax.debug.print("Gaussian x={} result={}\n center={} width={} amp={} base={}\n max_y={} min_y={}\n",
    #                 x, result, center, width, amplitude, baseline,
    #                 sampler_params['common']['max_y'], sampler_params['common']['min_y'])

    # Remove batch dimension if input x was originally 1D
    return result.squeeze()


def initialize_gaussian(key: chex.PRNGKey, params: EnvParams, action_dim: int) -> Dict:
    """
    Initializes Gaussian function parameters using config from EnvParams.

    Returns a nested dictionary including sampled parameters and calculated min/max y.
    """
    # Get Gaussian specific config from the main EnvParams
    gaussian_config = params.sampler_configs['specific']['gaussian']
    
    
    

    key_center, key_width, key_amp, key_base = jax.random.split(key, 4)
    dim = action_dim # Use concrete integer passed in
    lower, upper = params.x_range # General bounds from params
    opt_factor = gaussian_config.get('center_opt_factor', 1.0) # Factor for center range

    # Sample Gaussian parameters
    # Center (peak location) - sampled within a factor of the bounds
    center = jax.random.uniform(key_center, shape=(dim,),
                                minval=lower * opt_factor,
                                maxval=upper * opt_factor)

    # Width (controls steepness/std dev) - must be positive
    width = jax.random.uniform(key_width, shape=(dim,),
                               minval=gaussian_config['width_bounds'][0],
                               maxval=gaussian_config['width_bounds'][1])

    # Amplitude (height of peak above baseline) - must be positive
    amplitude = jax.random.uniform(key_amp, shape=(),
                                  minval=gaussian_config['amplitude_bounds'][0],
                                  maxval=gaussian_config['amplitude_bounds'][1])

    # Baseline (value far from center)
    baseline = jax.random.uniform(key_base, shape=(),
                                  minval=gaussian_config['baseline_bounds'][0],
                                  maxval=gaussian_config['baseline_bounds'][1])

    # --- Calculate max/min y for Gaussian ---
    optimum_point = center
    # Max y occurs exactly at the center (exponent = 0 -> exp(0) = 1)
    max_y = baseline + amplitude * 1.0

    # Min y occurs at the boundary point furthest from the center
    # Find coordinates of the furthest boundary point
    x_min_coords = jnp.where(jnp.abs(center - lower) > jnp.abs(upper - center), lower, upper)

    # Calculate the exponent term at this minimum point
    diff_min_sq = ((x_min_coords - center) / width) ** 2
    exponent_at_min = -0.5 * jnp.sum(diff_min_sq)

    # Calculate the minimum y value
    min_y_candidate = baseline + amplitude * jnp.exp(exponent_at_min)

    # Ensure min_y is distinct from max_y (optional safety)
    min_y = jnp.minimum(min_y_candidate, max_y - 1e-6)

    # --- Structure the output dictionary ---
    # Get the base structure, potentially with other sampler types
    sam_con = params.sampler_configs

    # Populate common parameters
    sam_con['common'] = {
        'type_index': 4, # Assign a unique index for Gaussian (e.g., 1)
        'optimum_point': optimum_point,
        'max_y': max_y,
        'min_y': min_y,
        'action_dim': dim,
    }

    # Populate Gaussian specific parameters
    # Ensure the 'gaussian' key exists if creating from scratch
    if 'gaussian' not in sam_con['specific']:
         sam_con['specific']['gaussian'] = {}

    sam_con['specific']['gaussian']['center'] = center
    sam_con['specific']['gaussian']['width'] = width
    sam_con['specific']['gaussian']['amplitude'] = amplitude
    sam_con['specific']['gaussian']['baseline'] = baseline
    
    
    # jax.debug.print("init_gauss {} min {}", params.sampler_configs['common']['type_index'], sam_con['common']['min_y'])

    # Add dummy/padding arrays for other sampler types if necessary
    # (similar to how you might pad the poly arrays)
    
    # print("gaussian", center.shape,"gsdfg")
    
    # jax.debug.print("gaussian x {} ",sam_con)
    
    

    return sam_con

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
        compute_y_neg_abs,  # Index 2
        compute_y_gaussian, # Index 3
        # Add other compute functions here...
    ]
    
    
    # branches = env_params.compute_switch

    num_samplers = len(branches)
    # Ensure type_index is treated as a JAX type for tracing if necessary
    type_index_jax = jnp.asarray(type_index)
    type_index_clipped = jnp.clip(type_index_jax, 0, num_samplers - 1)
    

    # Pass the full nested sampler_params dictionary
    y = jax.lax.switch(type_index_clipped, branches, x, sampler_params, env_params)
    # jax.debug.print("compute_y_sampler {} {} {}", type_index, x, y)

    return y



class FuncIndices(enum.Enum):
    ackley = enum.auto()    # Gets assigned 1
    poly = enum.auto()  # Gets assigned 2
    neg_abs = enum.auto() # Gets assigned 3
    gaussian = enum.auto()   # Gets assigned 4
    branin = enum.auto() # Gets assigned 5
    
class FunctionToSampler(enum.Enum):
    ackley = initialize_ackley
    poly = initialize_poly
    neg_abs = initialize_neg_abs
    gaussian = initialize_gaussian
    
    
func_to_sampler = {
    "ackley": initialize_ackley,
    "poly": initialize_poly,
    "neg_abs": initialize_neg_abs,
    "gaussian": initialize_gaussian,
}
    
func_to_compute = {
    "ackley": compute_y_ackley,
    "poly": compute_y_poly,
    "neg_abs": compute_y_neg_abs,
    "gaussian": compute_y_gaussian,
}
    
    
def function_params_to_dict(function: str, dim: int) -> Dict[str, Any]:
    if function == "ackley":
        return {
                "a": 15.0,
                "b": 0.1,
                "c": 2 * np.pi,
                'optimum_range_factor': 0.9,
                'a_bounds': (15.0, 20.0),
                'b_bounds': (0.1, 0.2),
                'c_bounds': (2 * np.pi, 2 * np.pi),
            }
           
    if function == "poly":
        return {
                'c': 10.0,
                'weights': jnp.full((dim,), jnp.nan, dtype=jnp.float32), # Dummy Array
                'x_max': jnp.full((dim,), jnp.nan, dtype=jnp.float32),   # Dummy Array
                'degree': 2,
                'steepness_factor': 1.0,
                'optimum_range_factor': 0.9,
                'c_bounds': (5.0, 20.0),
                'weight_bounds': (0.5, 2.0),
                'steep_min': 1.0,
                'steep_max': 2.0,
                
            }
    if function == "neg_abs":
        return {}
    if function == "gaussian":
        
        print("gaussian", jnp.full((dim,), jnp.nan, dtype=jnp.float32).shape)
        return {
                'center': jnp.full((dim,), jnp.nan, dtype=jnp.float32), # Dummy Array
                'width': jnp.full((dim,), jnp.nan, dtype=jnp.float32),  # Dummy Array
                'amplitude': jnp.nan,  # Dummy scalar
                'baseline': jnp.nan,   # Dummy scalar
                'center_opt_factor': 0.9,
                'width_bounds': (0.1, 2.0),
                'amplitude_bounds': (0.1, 2.0),
                'baseline_bounds': (-10.0, 10.0),
            }
      