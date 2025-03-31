import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np

from flax.linen.initializers import constant, orthogonal

class Flatten(nn.Module):
    def __call__(self, x):
        return x.reshape(x.shape[0], -1)

def flatten_repr_model():
    def thurn():
        return Flatten()
    return thurn

def atari_conv_repr_model():
    def thurn():
        return nn.Sequential([nn.Conv(32,
                                    kernel_size=(8, 8),
                                    strides=(4, 4),
                                    padding="VALID",
                                    kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0),
                                    ),nn.relu,
                            nn.Conv(
                                    64,
                                    kernel_size=(4, 4),
                                    strides=(2, 2),
                                    padding="VALID",
                                    kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0),
                                ),nn.relu,
                            nn.Conv(
                                    64,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding="VALID",
                                    kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0),
                                ),nn.relu,
                            ])
    return thurn


def mlp_repr_model(hidden_sizes=(256, 128)):
    """
    Creates an MLP-based representation model that extracts features
    from the input to be used by an LSTM.
    
    Args:
        hidden_sizes (tuple): Defines the sizes of the hidden layers.
    
    Returns:
        Function that initializes the MLP model.
    """
    def thurn():
        return nn.Sequential([
            nn.Dense(hidden_sizes[0], 
                     kernel_init=orthogonal(jnp.sqrt(2)), 
                     bias_init=constant(0.0)), 
            nn.relu,
            nn.Dense(hidden_sizes[1], 
                     kernel_init=orthogonal(jnp.sqrt(2)), 
                     bias_init=constant(0.0)), 
            nn.relu
        ])
    
    return thurn

# def dict_unpack_model(hidden_sizes=(256, 128)):
#     def thurn():
#         actions = 
#         return nn.Sequential([
#             nn.Dense(hidden_sizes[0], 
#                      kernel_init=orthogonal(jnp.sqrt(2)), 
#                      bias_init=constant(0.0)), 
#             nn.relu,
#             nn.Dense(hidden_sizes[1], 
#                      kernel_init=orthogonal(jnp.sqrt(2)), 
#                      bias_init=constant(0.0)), 
#             nn.relu
#         ])
    
#     return thurn


def dict_unpack_model(hidden_sizes=(256, 128)):
    """Actor model for continuous action spaces that outputs logits in a compatible format."""
    class ContinuousActor(nn.Module):
        @nn.compact
        def __call__(self, x):
            
          
            
            #x is a dictionary of inputs
            # keys = ["actions", "observations", "reward"]
            # actions.shape = (batch_size, action_dim)
            # observations.shape = (batch_size, 1)
            # reward.shape = (1,)
            # mask is a integer which says until which batch the data is valid, so if batch_size = 10 and mask = 5, then the first 5 batches are valid and the rest are invalid
           
            #This is to create a vmap for each field in the input
            def create_vmap_mlp():
                return nn.vmap(
                    nn.Sequential,  # individual networks
                    in_axes=0, out_axes=0,
                    variable_axes={'params': None},
                    split_rngs={'params': False}
                )

            # Dynamically create submodules, one for each field
            submodules = [create_vmap_mlp() for _ in x.keys()]


            #split the fields into batch related and step related
            batch_related = ["actions", "observations"]
            step_related = ["reward"]#, "best_action"]
            
            # print("x", x["actions"].shape)
            # Expand and the data for correct parsing through vmap
            expanded_inputs = [
                x[k][:, :, None, :]
                for k in batch_related
            ]
            
            
            

            # Process the batch related input through its corresponding MLP / Sequentual
            results = [
                module(
                    [nn.Dense(128),
                          nn.relu,
                          nn.Dense(64)]
                )(ins)
                for module, ins in zip(submodules, expanded_inputs)
            ]

            # Concatenate the results of the batch related data such that action and dimension are in the same axis per batch
            squeezed = jnp.squeeze(jnp.array(results), axis=3) # shape (2, batch_size, hidden_size)
            
            batch_related_data = jnp.transpose(squeezed, (1, 2, 0, 3)) #shape (batch_size, 2, hidden_size)
            flattened_batch = batch_related_data.reshape((batch_related_data.shape[0], batch_related_data.shape[1], -1)) # shape (batch_size, 2 * hidden_size)
            # print("eyno", flattened_batch.shape)
            
            
            #vmap over batch related data such that action and observation can have learned correlation
            action_mapping = nn.vmap(  
                nn.Sequential,  # share networks
                in_axes=0, out_axes=0,
                variable_axes={'params': None},
                split_rngs={'params': False},
            )

            expanded_input = action_mapping(
                    [nn.Dense(256),
                          nn.relu,
                          nn.Dense(128)]
                )(flattened_batch)
            #otuput shape (batch_size, hidden_size (5))
            
            ####missing step for batch learning
            # can sum over batch axis to get a single representation for the batch or take mean max, or any other aggregation / pooling but needs to be capable of dealing with masked batches
            # output should be of shape (1, hidden_size) / (hidden_size,)
            
            
            
            expanded_input = jnp.mean(expanded_input, axis=1)
            
            
            
            

            # conc_step  = jnp.concatenate([x[key] for key in step_related], axis=1)
            step_expans = nn.Sequential([nn.Dense(128),
                          nn.relu,
                          nn.Dense(256)])
            

            #expand the step related data
            step_related_data = step_expans(x["reward"]) # shape (6,)
            
            final_represenation_layer = nn.Sequential([nn.Dense(256),
                          nn.relu,
                          nn.Dense(256)]) 
            
            #combine the step related data with the batch related data
            batch_with_step = jnp.concatenate([expanded_input, step_related_data], axis=1)
            final_output = final_represenation_layer(batch_with_step)
            # final_output = jnp.expand_dims(final_output, axis=0)
            return final_output
    
    return lambda: ContinuousActor()



def dict_unpack_mask(
    batch_expand_hidden=(64, 64),  # For processing individual fields
    batch_combine_hidden=(64, 5),  # For processing combined action/observation
    step_expand_hidden=(64, 64),  # For processing reward
    input_combine_hidden=(64, 64)  # For final representation
):
    """Actor model for continuous action spaces with parameterizable layer sizes."""
    class ContinuousActor(nn.Module):
        @nn.compact
        def __call__(self, x):
            
            # Function to create a sequential MLP with parameterized hidden sizes
            def create_mlp_layers(hidden_sizes):
                layers = []
                for size in hidden_sizes[:-1]:
                    layers.append(nn.Dense(size))
                    layers.append(nn.relu)
                # Last layer without activation
                layers.append(nn.Dense(hidden_sizes[-1]))
                return layers
            
            # Create vmap for each field in the input
            def create_vmap_mlp():
                return nn.vmap(
                    nn.Sequential,  # individual networks
                    in_axes=0, out_axes=0,
                    variable_axes={'params': None},
                    split_rngs={'params': False}
                )
                
            batch_related = ["actions", "observations"]
            step_related = ["reward"]

            # Dynamically create submodules, one for each field
            submodules = [create_vmap_mlp() for _ in batch_related]

            # Split the fields into batch related and step related
            
            
            # for k in x.keys():
            #     print(k, x[k])
            
            
            
            # Expand the data for correct parsing through vmap
            expanded_inputs = [
                x[k][:, :, None, :]
                for k in batch_related
            ]
            
           
            
            # Process the batch related input through its corresponding MLP / Sequential
            results = [
                module(create_mlp_layers(batch_expand_hidden))(ins)
                for module, ins in zip(submodules, expanded_inputs)
            ]
           
            

            # Concatenate the results of the batch related data
            squeezed = jnp.squeeze(jnp.array(results), axis=3)  # shape (2, batch_size, hidden_size)
            
            
            
            batch_related_data = jnp.transpose(squeezed, (1, 2, 0, 3))  # shape (batch_size, 2, hidden_size)
            flattened_batch = batch_related_data.reshape((batch_related_data.shape[0], batch_related_data.shape[1], -1))
            
            
            # vmap over batch related data such that action and observation can have learned correlation
            action_mapping = nn.vmap(  
                nn.Sequential,  # share networks
                in_axes=0, out_axes=0,
                variable_axes={'params': None},
                split_rngs={'params': False},
            )

            expanded_input = action_mapping(
                create_mlp_layers(batch_combine_hidden)
            )(flattened_batch)
            
            
            # Handle masked batches
            mask = jnp.asarray(x["mask"], dtype=jnp.int32)
            batch_averaged = jnp.sum(expanded_input, axis=1)
            batch_normalised = jnp.divide(batch_averaged, mask)
            
            
            
            # Process step-related data
            step_expans = nn.Sequential(create_mlp_layers(step_expand_hidden))
            step_related_data = step_expans(x["reward"])
            
            
            # Combine the step related data with the batch related data
            batch_with_step = jnp.concatenate([batch_normalised, step_related_data], axis=1)
            
            # Final representation layer
            final_represenation_layer = nn.Sequential(create_mlp_layers(input_combine_hidden))
            final_output = final_represenation_layer(batch_with_step)
            
            return final_output
    
    return lambda: ContinuousActor()


def simple_mlp(
    hidden_sizes=(256, 128)
):
    """Actor model for continuous action spaces with parameterizable layer sizes."""
    class ContinuousActor(nn.Module):
        @nn.compact
        def __call__(self, x):
            
            
            
            # Expand the data for correct parsing through vmap
            expanded_inputs = [
                jnp.reshape(x[k], (x[k].shape[0], 1))
                for k in ["actions", "observations", "reward"]
            ]
            
            
            
            
            conc_input = jnp.concatenate(expanded_inputs, axis=-1)
            # jax.debug.print("conc_input {}\n {}", expanded_inputs, conc_input)
            # print("conc_input", conc_input)
            # jax.debug.print("conc_input {}\n", conc_input)
            # print("expanded_inputs", expanded_inputs[0].shape)
            # print("act", x["actions"].shape)
            
            model = nn.Sequential([
            nn.Dense(hidden_sizes[0], 
                     kernel_init=orthogonal(jnp.sqrt(2)), 
                     bias_init=constant(0.0)), 
            nn.relu,
            nn.Dense(hidden_sizes[1], 
                     kernel_init=orthogonal(jnp.sqrt(2)), 
                     bias_init=constant(0.0)), 
            nn.relu
            ])
            
            out = model(conc_input)
            # out = jnp.expand_dims(out, axis=0)
            # print("out", out.shape)
            # print("out", out.shape, x["actions"].shape)
    
            
            return out
    
    return lambda: ContinuousActor()
