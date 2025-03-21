import flax.linen as nn
import jax.numpy as jnp
import jax

def planar_flow(a_dim):
    """Actor model for continuous action spaces with parameterizable layer sizes.
    
    Args:
        shared_hidden_sizes: Tuple of hidden layer sizes for the shared network
        policy_hidden_sizes: Tuple of hidden layer sizes for both mean and log_std networks,
                            with the last value being used as the action dimension
    
    Returns:
        A function that returns a ContinuousActor module
    """
    # Extract action dimension from the last element of policy_hidden_sizes
    class PlanarFlowStep(nn.Module):
        num_flows: int= 4
        
        @nn.compact
        def __call__(self, x):
            batch_shape = x.shape[:-1]
            input_dim = x.shape[-1]
            log_det = jnp.zeros(batch_shape)
            
           
            
            print("as os this shaped", x.shape)
            # x shape: [S, a_dim] - samples for a single batch element
            sample_shape = x.shape[:-1]  # [S]
            input_dim = x.shape[-1]      # a_dim
            log_det = jnp.zeros(sample_shape)
            
            for i in range(self.num_flows):
                # Parameters for the flow
                u = self.param(f'u_{i}', nn.initializers.normal(0.01), (input_dim,))
                w = self.param(f'w_{i}', nn.initializers.normal(0.01), (input_dim,))
                b = self.param(f'b_{i}', nn.initializers.zeros, (1,))
                
                # Apply constraints (if needed)
                wu = jnp.dot(w, u)
                m = -1 + jax.nn.softplus(wu)
                u_hat = u + (m - wu) * w / (jnp.dot(w, w) + 1e-8)
                
                # Apply the transformation
                wx_b = jnp.einsum('sd,d->s', x, w) + b  # [S, a_dim] dot [a_dim] -> [S]
                z = x + u_hat * jax.nn.tanh(wx_b)[:, None]  # Add broadcast dimension
                
                # Compute log determinant
                phi = (1 - jnp.tanh(wx_b)**2)[:, None] * w  # [S, 1] * [a_dim] -> [S, a_dim]
                psi = jnp.einsum('sd,d->s', phi, u_hat)  # [S, a_dim] dot [a_dim] -> [S]
                flow_log_det = jnp.log(jnp.abs(1 + psi) + 1e-8)
                log_det = log_det + flow_log_det
                
                # Update x for the next flow
                x = z
                
            print("outputs shape", x, "total_log_det shape", log_det.shape)
            return x, log_det
        
    return lambda: PlanarFlowStep()
        
def autoregressive_iaf_flow(a_dims):
    class ARNet(nn.Module):
        a_dim: int
        hidden_size: int = 32  # you can adjust hidden layer size as needed
        
        @nn.compact
        def __call__(self, context):
            # A simple 2-layer MLP that produces shift and log-scale parameters.
            h = nn.Dense(self.hidden_size)(context)
            h = nn.tanh(h)
            mu = nn.Dense(self.a_dim)(h)
            log_sigma = nn.Dense(self.a_dim)(h)
            return mu, log_sigma

    class IAFStep(nn.Module):
        num_flows: int = 3  # number of flow steps to apply
        
        @nn.compact
        def __call__(self, x):
            # x shape: [S, a_dim]
            S, a_dim = x.shape
            total_log_det = jnp.zeros(())  # scalar accumulation of log-determinants
            outputs = x  # initialize outputs as the input
            
            # For each flow step, we apply an autoregressive transformation over samples
            for flow in range(self.num_flows):
                # Define an AR network (could be shared or flow-specific)
                arn = ARNet(a_dim, name=f"arn_{flow}")
                context = jnp.zeros((a_dim,))  # initial context for this flow step
                new_outputs = []  # will collect transformed samples for this flow step
                
                # Process samples in order to impose autoregressive dependency
                for i in range(S):
                    # Compute transformation parameters conditioned on the current context.
                    mu, log_sigma = arn(context)
                    sigma = jnp.exp(log_sigma)
                    
                    # Apply the affine transformation: z_i = sigma * x_i + mu
                    z_i = sigma * outputs[i] + mu
                    
                    # Update the total log-determinant (sum over a_dim for the current sample)
                    total_log_det += jnp.sum(log_sigma)
                    
                    # Update context with the transformed sample.
                    # (Here we simply set the context to z_i; more sophisticated updates are possible.)
                    context = z_i
                    new_outputs.append(z_i)
                
                # Stack the transformed samples back into a tensor of shape [S, a_dim]
                outputs = jnp.stack(new_outputs, axis=0)
                
            print("outputs shape", outputs.shape, "total_log_det shape", total_log_det.shape)
            return outputs, total_log_det
        
        

    return lambda: IAFStep()

# Example usage:
#   iaf_flow = AutoregressiveIAFFlow(a_dim=action_dim, hidden_dim=32)
#   transformed, total_log_det = iaf_flow.apply({'params': params}, raw_actions)

        
        
    
            
    
    # class PlanarFlowStep(nn.Module):
  

    #     @nn.compact
    #     def __call__(self, sample):
    #         """
    #         Applies a planar flow transformation to one sample vector.
    #         The transformation is:
    #             z' = z + u * h(w^T z + b + conditioning)
    #         where h = tanh.
    #         We use the previous carry (summarized via a Dense layer) as a conditioning term.
    #         """
    #         # Use carry to produce a conditioning term.
    #         # For the first sample, carry will be zeros.
    #         conditioning = nn.Dense(a_dim, name="cond_dense")(sample)  # shape: (a_dim,)
    #         cond_scalar = jnp.mean(conditioning)  # simple summary (could be more complex)
            
    #         # Flow parameters (learned per step)
    #         u = self.param('u', nn.initializers.normal(), (a_dim,))
    #         w = self.param('w', nn.initializers.normal(), (a_dim,))
    #         b = self.param('b', nn.initializers.zeros, ())  # scalar bias

    #         # Compute linear transformation with conditioning
    #         linear = jnp.dot(sample, w) + b + cond_scalar  # scalar per sample
    #         h = jnp.tanh(linear)
    #         transformed = sample + u * h  # broadcast h (scalar) over a_dim
            
    #         # Compute derivative: h'(linear) = 1 - tanh(linear)^2
    #         h_prime = 1.0 - jnp.tanh(linear) ** 2
    #         psi = h_prime * w  # shape: (a_dim,)
    #         u_psi = jnp.dot(u, psi)  # scalar
    #         log_det = jnp.log(jnp.abs(1.0 + u_psi) + 1e-6)  # add small constant for stability
            
    #         # Update carry with the transformed sample (or use a more complex update)
    #         new_carry = transformed
    #         return new_carry, (transformed, log_det)
        
    return lambda: PlanarFlowStep()