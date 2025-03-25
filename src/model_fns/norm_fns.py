import flax.linen as nn
import jax.numpy as jnp
import jax

def planar_flow(a_dim):
    """
    Returns a callable that constructs a single planar flow step.
    
    Input:
      x: [S, a_dim] – S samples (e.g. in your GMM code, S could be your batch dimension for one environment)
      
    Transformation:
      z' = z + u_hat * tanh(wᵀz + b)
      
      Here u, w ∈ ℝ^(a_dim) and b ∈ ℝ are learnable parameters.
      To ensure invertibility we reparameterise u into u_hat:
          u_hat = u + ((m - wᵀu) * w) / (||w||² + ε)
      where m = -1 + softplus(wᵀu).
      
    Log–determinant:
      For each sample i we compute
          ψ(z_i) = (1 - tanh(wᵀz_i + b)²) * w,
      and then the Jacobian for that sample is:
          det(J) = 1 + u_hatᵀ ψ(z_i)
      We take the log–absolute value and finally sum over S to get a scalar.
    """
    class PlanarFlowStep(nn.Module):
        @nn.compact
        def __call__(self, x):
            # x shape: [S, a_dim]
            # Initialize parameters:
            u = self.param("u", nn.initializers.normal(0.01), (a_dim,))
            w = self.param("w", nn.initializers.normal(0.01), (a_dim,))
            b = self.param("b", nn.initializers.zeros, ())
            
            # --- Invertibility fix: compute u_hat ---
            # Compute dot product: scalar = wᵀ u.
            w_dot_u = jnp.dot(w, u)
            # m = -1 + softplus(wᵀu)
            m = -1.0 + jax.nn.softplus(w_dot_u)
            # Avoid division by zero:
            w_norm_sq = jnp.sum(w**2) + 1e-8
            # u_hat is used in place of u for the transformation.
            u_hat = u + ((m - w_dot_u) * w) / w_norm_sq

            # --- Compute the transformation ---
            # inner = wᵀx + b, computed for each sample; shape: [S]
            inner = jnp.dot(x, w) + b
            # Apply nonlinearity: h(inner); here tanh is used.
            h = jnp.tanh(inner)  # shape: [S]
            # Transform each sample: x + u_hat * h. We expand h to [S, 1].
            transformed = x + jnp.expand_dims(h, -1) * u_hat  # shape: [S, a_dim]
            
            # --- Compute log–determinant ---
            # Derivative of tanh: h'(inner) = 1 - tanh(inner)^2.
            psi = (1 - jnp.tanh(inner)**2)  # shape: [S]
            # For each sample, compute u_hatᵀ * (h'(inner)*w).
            # Here, we “broadcast” psi (shape [S]) to multiply with w (shape [a_dim]) to get [S, a_dim],
            # and then take the dot product with u_hat (of shape [a_dim]) for each sample.
            dot = jnp.sum(u_hat * (psi[:, None] * w), axis=-1)  # shape: [S]
            # The Jacobian determinant for each sample is: 1 + dot.
            # Take log–abs (with an epsilon for numerical stability).
            log_det = jnp.log(jnp.abs(1 + dot) + 1e-8)  # shape: [S]
            # Sum over all S samples to return a single scalar log–determinant for this flow step.
            # total_log_det = jnp.sum(log_det)
            return transformed, log_det
    # Return a callable (using a lambda) so that you can instantiate the module as in your code.
    return lambda: PlanarFlowStep()
        
def autoregressive_flow(a_dim, hidden_size=32):
    """
    Returns a callable that constructs a single autoregressive flow step.
    ...
    """
    class AutoregressiveFlowStep(nn.Module):
        @nn.compact
        def __call__(self, x):
            # x shape: [S, a_dim]
            lstm_cell = nn.LSTMCell(features=hidden_size)
            # Initialize the LSTM state (carry) with batch size 1.
            init_carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (1, a_dim))

            
            # Expand x to include a batch dimension.
            x_seq = jnp.expand_dims(x, axis=1)  # shape: [S, 1, a_dim]
            
            def step_fn(carry, x_t):
                # x_t shape: [1, a_dim]
                h = carry[1]  # hidden state: shape [1, hidden_size]
                params = nn.Dense(features=2 * a_dim)(h)  # shape: [1, 2*a_dim]
                mu, log_sigma = jnp.split(params, 2, axis=-1)  # each: [1, a_dim]
                sigma = jnp.exp(log_sigma)
                y_t = sigma * x_t + mu  # shape: [1, a_dim]
                log_det_t = jnp.sum(log_sigma, axis=-1)  # shape: [1]
                new_carry, _ = lstm_cell(carry, y_t)
                return new_carry, (y_t, log_det_t)
            
            final_carry, outputs = jax.lax.scan(step_fn, init_carry, x_seq)
            y_seq, log_det_seq = outputs  # y_seq: [S, 1, a_dim], log_det_seq: [S, 1]
            y_seq = jnp.squeeze(y_seq, axis=1)      
            log_det_seq = jnp.squeeze(log_det_seq, axis=1)
            return y_seq, log_det_seq
    return lambda: AutoregressiveFlowStep()
