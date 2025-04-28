# import jax
# import jax.numpy as jnp
# import chex
# import flax.struct as struct # Or use standard dataclasses
# from gymnax.environments import environment, spaces
# from typing import Tuple, Optional, Dict, Any
# from src.tasks.envs.jax_env.jax_function_samplers import initialize_ackley, initialize_poly

# import enum

# # --- Option 1: Using enum.Enum with automatic integer values ---
# # Enum members get assigned unique integer values starting from 1 by default.
# class FuncIndices(enum.Enum):
#     ackley = enum.auto()    # Gets assigned 1
#     poly = enum.auto()  # Gets assigned 2
#     cosine = enum.auto()   # Gets assigned 3
#     branin = enum.auto() # Gets assigned 4
    
# def function_params_to_dict(function: str, dim: int) -> Dict[str, Any]:
#     if function == "ackley":
#         return {
#                 "a": 0.2,
#                 "b": 0.2,
#                 "c": 0.2,
#             }
           
#     if function == "poly":
#         return {
#                 'c': jnp.nan,
#                 'weights': jnp.full((dim,), jnp.nan, dtype=jnp.float32), # Dummy Array
#                 'x_max': jnp.full((dim,), jnp.nan, dtype=jnp.float32),   # Dummy Array
#                 'degree': 0,
#             }
      
        

# @struct.dataclass
# class EnvParams:
#     """Static environment parameters.
#        Derived values like max_steps_in_episode should be calculated
#        *before* instantiation and passed in.
#     """
#     # --- Fields involved in calculation (must be provided at init) ---
#     total_samples: int
#     max_batches: int

#     # --- Derived field (now init=True, value passed during init) ---
#     max_steps_in_episode: int # No default, must be provided
#     sampler_configs: Dict[str, Dict]
#     # --- Other fields ---
#     x_range: Tuple[float, float] = (-5.0, 5.0)
#     # Provide defaults using default_factory or make them required too
#     batches: chex.Array = struct.field(default_factory=lambda: jnp.array([1, 1]))
#     action_dim: int = 2 # Example default
#     use_random_action_on_reset: bool = False
#     use_random_action_on_step: bool = False
#     function_type_indices: chex.Array = struct.field(default_factory=lambda: jnp.arange(2))
#     r_scale: float = 10.0
#     r_best: float = 0.8
#     r_impr: float = 0.1
#     r_new_best: float = 0.1
#     r_obs: float = 0.0
#     r_mse: float = 0.0
#     r_suc: float = 3.0
#     success_threshold: float = 0.95


#    # NO __post_init__ needed for this calculation anymore
# def create_env_params(config) -> EnvParams:
#     """Factory function to create EnvParams."""
#     # Calculate max_steps_in_episode based on total_samples and max_batches
#     m_batches = int(jnp.max(jnp.array(config["batches"])))
#     max_steps_in_episode = config["total_episode_samples"] // m_batches
#     print("splitting of batches !!!!!!!!!")
    
    
#     fun_ind = [FuncIndices[fun].value - 1 for fun in config["function_types"]]
    
    
    
#     sampler_params = {
#         'common': {
#             'type_index': fun_ind[0], # Assuming Ackley is index 0
#             'optimum_point': 0,
#             'max_y': 0,
#             'min_y': 0,
#             'action_dim': config["action_dim"],
#         },
#         'specific': {
#         }
#     }
    
    
    
#     for function in config["function_types"]:
#             sampler_params['specific'][function] = function_params_to_dict(function, config["action_dim"])
      
#     print("function types", config["function_types"], fun_ind, sampler_params)

   
#     return EnvParams(
#         total_samples=config["total_episode_samples"],
#         batches=jnp.array(config["batches"]),#this gives an error how can I make this correct
#         max_batches=m_batches,
#         max_steps_in_episode=max_steps_in_episode,
#         sampler_configs=sampler_params,
#         x_range=config["bounds"],
#         action_dim=config["action_dim"],
#         use_random_action_on_step=config["random"],
#         function_type_indices=jnp.array(fun_ind),
#         # ... Add any other default values needed ...
#     )
    
    
