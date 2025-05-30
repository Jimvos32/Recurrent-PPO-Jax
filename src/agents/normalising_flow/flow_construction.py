import jax
import jax.random as jr
import inspect
from flowjax.bijections import AbstractBijection, Chain, Planar, BlockAutoregressiveNetwork, Affine
from typing import List, Tuple, Dict, Any, Type, Optional, Callable



BIJECTION_REGISTRY: Dict[str, Type[AbstractBijection]] = {
    "Planar": Planar,
    "BlockAutoregressiveNetwork": BlockAutoregressiveNetwork,
    "Affine": Affine,
    # Add other bijections from flowjax or custom ones here
    # e.g., "MaskedAutoregressive": flowjax.bijections.MaskedAutoregressive,
}

ACTIVATION_REGISTRY: Dict[str, Callable] = {
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "leaky_relu": jax.nn.leaky_relu,
    # Add other JAX activation functions as needed
}

def create_flow_chain(
        key: jax.random.PRNGKey,
        action_dim: int,
        layer_configs: Optional[List[Tuple[Type[AbstractBijection], Dict[str, Any]]]]
    ) -> Optional[Chain]:
        """
        Initializes a sequence of flowjax bijections and combines them into a Chain.

        Args:
            key: JAX PRNGKey for initializing the layers.
            action_dim: The dimension of the space the flow operates on.
            layer_configs: A list of tuples, where each tuple contains:
                - The bijection class for the layer.
                - A dictionary of keyword arguments for that layer's __init__.
                Returns None if layer_configs is None or empty.

        Returns:
            An initialized flowjax.bijections.Chain object, or None if no layers were configured.
        """
        if not layer_configs:
            print("No flow layer configurations provided. Skipping flow chain creation.")
            return None

        flow_layers = []
        # Split the main key for each layer that might need initialization
        layer_keys = jr.split(key, len(layer_configs))

        for i, (layer_class, layer_kwargs) in enumerate(layer_configs):
            layer_key_for_this_layer = layer_keys[i]
            
            # Inspect the __init__ signature of the layer class
            try:
                init_signature = inspect.signature(layer_class.__init__)
                init_param_names = list(init_signature.parameters.keys())
            except TypeError: # Happens for some built-in types or non-Python functions
                init_param_names = []
                print(f"Warning: Could not inspect signature for {layer_class.__name__}. Assuming no special params like 'key' or 'dim'.")

            # Start with a copy of user-provided kwargs for the current layer
            current_layer_init_args = dict(layer_kwargs) 
            
            # Conditionally add the 'key' if it's in the signature and not already provided by user config
            if 'key' in init_param_names and 'key' not in current_layer_init_args:
                current_layer_init_args['key'] = layer_key_for_this_layer
            
            # Dimension handling:
            # Try to set the primary dimension for the bijection if not already provided in layer_kwargs.
            # Prioritize 'dim', then 'shape', then 'n_features'.
            dim_param_set = False
            if 'dim' in init_param_names:
                if 'dim' not in current_layer_init_args: # Only set if not provided by user
                    current_layer_init_args['dim'] = action_dim
                dim_param_set = True # Mark that 'dim' was considered (either set or user-provided)
            elif 'shape' in init_param_names: 
                if 'shape' not in current_layer_init_args: # Only set if not provided by user
                    current_layer_init_args['shape'] = (action_dim,)
                dim_param_set = True # Mark that 'shape' was considered
            elif 'n_features' in init_param_names:
                if 'n_features' not in current_layer_init_args: # Only set if not provided by user
                    current_layer_init_args['n_features'] = action_dim
                dim_param_set = True # Mark that 'n_features' was considered
            
            # If no standard dimension parameter was applicable or set by the above logic
            if not dim_param_set:
                # This warning triggers if none of 'dim', 'shape', or 'n_features' are in the signature.
                # If they are in the signature but also in layer_kwargs, dim_param_set would be true.
                is_any_known_dim_param_in_signature = any(p in init_param_names for p in ['dim', 'shape', 'n_features'])
                if not is_any_known_dim_param_in_signature:
                    print(f"Warning: {layer_class.__name__} does not accept common dimension parameters ('dim', 'shape', 'n_features'). "
                        f"Ensure dimensioning is handled correctly if required (e.g. via other params in config). "
                        f"Current action_dim is {action_dim}. Available init params: {init_param_names}")


            try:
                # print(f"Initializing {layer_class.__name__} with args: {current_layer_init_args}")
                layer_instance = layer_class(**current_layer_init_args)
                flow_layers.append(layer_instance)
            except Exception as e:
                import traceback
                # print(f"Error initializing {layer_class.__name__} with args: {current_layer_init_args}")
                traceback.print_exc() 
                raise e

        if flow_layers:
            flow_chain = Chain(flow_layers)
            return flow_chain
        else:
            print("No flow layers were successfully initialized.")
            return None
        
    
    
def parse_hydra_flow_config(
    hydra_config_layers: List[Dict[str, Any]],
    # action_dim is not strictly needed here if create_flow_chain handles it,
    # but can be useful for parameter validation or complex setups.
) -> List[Tuple[Type[AbstractBijection], Dict[str, Any]]]:
    """
    Parses a list of layer configurations (e.g., from a Hydra config)
    and prepares it for the create_flow_chain function.

    Args:
        hydra_config_layers: A list of dictionaries, where each dictionary
                             represents a flow layer and has:
                             - "type": str, the name of the bijection class (e.g., "Planar").
                             - "params": Dict[str, Any], parameters for the bijection's __init__.

    Returns:
        A list of tuples in the format expected by create_flow_chain.
    """
    parsed_layer_configs = []

    for layer_conf in hydra_config_layers:
        bijection_type_str = layer_conf.get("type")
        if not bijection_type_str:
            raise ValueError("Each layer configuration must have a 'type' field.")

        BijectionClass = BIJECTION_REGISTRY.get(bijection_type_str)
        if not BijectionClass:
            raise ValueError(f"Unknown bijection type: '{bijection_type_str}'. "
                             f"Available types: {list(BIJECTION_REGISTRY.keys())}")

        params = layer_conf.get("params", {}).copy() # Get params, default to empty dict

        # Handle special parameter conversions, like activation functions
        if "activation" in params and isinstance(params["activation"], str):
            activation_str = params["activation"]
            activation_fn = ACTIVATION_REGISTRY.get(activation_str)
            if not activation_fn:
                raise ValueError(f"Unknown activation function: '{activation_str}'. "
                                 f"Available activations: {list(ACTIVATION_REGISTRY.keys())}")
            params["activation"] = activation_fn
        
        # For BlockAutoregressiveNetwork, 'conditioner' might be complex.
        # Here we assume it's either pre-configured or parameters for AutoregressiveMLP
        # are passed directly if BlockAutoregressiveNetwork uses it internally.
        # If 'conditioner' itself needs to be constructed (e.g. an AutoregressiveMLP),
        # that logic would go here.
        # For example, BlockAutoregressiveNetwork's default conditioner is AutoregressiveMLP.
        # Its parameters (hidden_dims, activation, etc.) can be directly in `params`.

        # `key` and `dim`/`shape` will be handled by `create_flow_chain`.
        # We just pass through the other specific parameters for the bijection.

        parsed_layer_configs.append((BijectionClass, params))

    return parsed_layer_configs