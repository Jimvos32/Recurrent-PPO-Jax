import jax
import jax.random as jr
import inspect
from flowjax.bijections import AbstractBijection, Chain
from typing import List, Tuple, Dict, Any, Type, Optional

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
            - A dictionary of keyword arguments for that layer's __init__
              (excluding 'key' and the primary dimension argument).
            Returns None if layer_configs is None or empty.

    Returns:
        An initialized flowjax.bijections.Chain object, or None if no layers were configured.
    """
    if not layer_configs:
        print("No flow layer configurations provided. Skipping flow chain creation.")
        return None

    flow_layers = []
    layer_keys = jr.split(key, len(layer_configs))

    for i, (layer_class, layer_kwargs) in enumerate(layer_configs):
        layer_key = layer_keys[i]
        
        # Start with key and user-provided kwargs for the current layer
        current_layer_init_args = {'key': layer_key, **layer_kwargs}
        
        
        # Inspect the __init__ signature of the layer class
        # to determine the correct parameter name for the data dimension.
        init_param_names = list(inspect.signature(layer_class.__init__).parameters.keys())

        # Try to set the primary dimension for the bijection
        dimension_set = False
        if 'dim' in init_param_names:
            current_layer_init_args['dim'] = action_dim
            dimension_set = True
        elif 'shape' in init_param_names:
            # Elementwise bijections often take a tuple for shape, e.g., (action_dim,)
            current_layer_init_args['shape'] = (action_dim,)
            dimension_set = True
        elif 'n_params' in init_param_names and 'dim' not in init_param_names and 'shape' not in init_param_names:
            # Fallback if 'n_params' is the only likely candidate for the primary data dimension
            current_layer_init_args['n_params'] = action_dim
            dimension_set = True
        


        try:
            layer_instance = layer_class(**current_layer_init_args)
            flow_layers.append(layer_instance)
        except Exception as e:
            import traceback
            traceback.print_exc() # Print full traceback for debugging initialization errors
            raise e

    if flow_layers:
        flow_chain = Chain(flow_layers)
        # The shape of the chain is the shape of its first bijection.
        # It should match (action_dim,) for unbatched data.
        expected_shape = (action_dim,)
      
        return flow_chain
    else:
        print("No flow layers were successfully initialized.")
        return None