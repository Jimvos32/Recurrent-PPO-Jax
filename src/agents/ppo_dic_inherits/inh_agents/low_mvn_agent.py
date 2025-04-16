import jax, jax.numpy as jnp
from src.agents.ppo_dic_inherits.inh_agents.standard_sampling import SamplingImplBase
from functools import partial


class LowRankMVN(SamplingImplBase):
    def __init__(self, max_batch, action_dim, sampling_distribution):
    
        super().__init__(max_batch, action_dim, sampling_distribution)
        self.rank_k = sampling_distribution         # k
        self.epsilon = 1e-6
        
       

        
        self.flat_action_dim = self.action_dim * self.batch_size  # Flattened action dimension for the network output
        # Calculate the expected size of the network's output logits
        self.expected_logit_size = self.flat_action_dim * (2 + self.rank_k)
        print(f"LowRankMVNSampling initialized: s={self.batch_size}, a={self.action_dim}, k={self.rank_k}")
        print(f"Flattened action dimension: {self.flat_action_dim}")
        print(f"Expected network output size per action: {self.expected_logit_size}")

    
    def sampling_differ(self, act_logits, key, masks):
        """
        Samples actions using a Low-Rank Multivariate Normal distribution.

        Args:
            act_logits: Raw output tensor from the policy network.
                        Expected shape [N, expected_logit_size], where N is the
                        number of parallel environments (JAX batch dim).
            key: JAX random key.
            masks: Optional padding mask (structure depends on your usage).

        Returns:
            Sampled actions tensor of shape [N, s, a] (or [N, batch_size, action_dim]).
        """
        # N is the number of parallel environments (outer batch dimension)
        N = act_logits.shape[0]

        # --- Sanity Check: Verify network output size ---
        if act_logits.shape[-1] != self.expected_logit_size:
             raise ValueError(f"Network output dimension mismatch! "
                              f"Expected act_logits last dim {self.expected_logit_size}, "
                              f"but got {act_logits.shape[-1]}. Check network definition.")

        # --- 1. Split Network Output into mu, log_std, V ---
        # Mean vector mu: shape [N, s*a]
        
        mu = jnp.squeeze(act_logits[..., :self.flat_action_dim], axis=1)

        # Log standard deviations for diagonal part: shape [N, s*a]
        log_std = jnp.squeeze(act_logits[..., self.flat_action_dim : 2 * self.flat_action_dim], axis=1)

        # Low-rank factor V (flattened): shape [N, s*a*k]
        V_flat = act_logits[..., 2 * self.flat_action_dim :]
        # Reshape V to its proper matrix form: shape [N, s*a, k]
        V = jnp.reshape(V_flat, (N, self.flat_action_dim, self.rank_k))

        # print(f"mu shape: {mu.shape}, log_std shape: {log_std.shape}, V shape: {V.shape}")
        # --- 2. Prepare Distribution Parameters ---
        # Calculate standard deviations for the diagonal part (ensure positivity)
        # sqrt_D_diag: shape [N, s*a]
        sqrt_D_diag = jnp.exp(log_std)

        # --- 3. Generate Noise ---
        # Split the key for independent noise samples
        key_diag, key_rank = jax.random.split(key)

        # Noise for the diagonal part (standard normal)
        # z_diag: shape [N, s*a]
        z_diag = jax.random.normal(key_diag, shape=(N, self.flat_action_dim))

        # Noise for the low-rank part (standard normal)
        # z_rank: shape [N, k]
        z_rank = jax.random.normal(key_rank, shape=(N, self.rank_k))

        # --- 4. Compute Sample using Low-Rank MVN Formula ---
        # action = mu + sqrt(D) * z_diag + V @ z_rank
        # where sqrt(D) applies element-wise multiplication here.

        # Calculate diagonal term contribution: shape [N, s*a]
        diag_contribution = sqrt_D_diag * z_diag

        # Calculate low-rank term contribution using batch matrix-vector product
        # V: [N, s*a, k], z_rank: [N, k]
        # einsum sums over the rank dimension 'k' -> shape [N, s*a]
        rank_contribution = jnp.einsum('Nak,Nk->Na', V, z_rank)
        # Alternative using matmul (might be less explicit):
        # rank_contribution = jnp.matmul(V, z_rank[..., None])[..., 0]

        # Combine terms to get the flat action sample (pre-transformation)
        # action_flat: shape [N, s*a]
        action_flat = mu + diag_contribution + rank_contribution

        # --- 5. Reshape and Apply Final Transformation (tanh) ---
        # Reshape from flat [N, s*a] to the desired joint action shape [N, s, a]
        acts_tick = jnp.reshape(action_flat, (N, self.batch_size, self.action_dim))

        
        acts_tick = jnp.tanh(acts_tick)
        
        # You might need 'action_pre_tanh' later for the log_prob calculation
        # if you implement the tanh correction.
    

        # --- 6. Apply Masking (Optional) ---
        # The logic here depends on how your masks are defined and used.
        # Adapt this section based on your apply_padding_mask implementation.
        if masks is not None:
            # Example: If mask just indicates valid samples in the 's' dimension [N, s]
            # if masks.ndim == 2 and masks.shape == (N, self.batch_size):
            #     # Broadcast mask to [N, s, a] and apply padding where mask is False
            #     acts_tick = jnp.where(masks[:, :, None], acts_tick, padding_value)
            # else: # Assume mask is [N, s, a] or handled by the function
            acts_tick = self.apply_padding_mask(acts_tick, masks) # Use your existing function

        # Store pre-tanh actions if needed for log prob calculation (optional)
        # You could return both or store it in the class instance if needed later.
        # self._last_pre_tanh_action = action_pre_tanh

        return acts_tick # Shape [N, s, a]

    # # --- Placeholder for Log Probability ---
    # def gaussian_log_prob(self, actions, act_logits):
    #     """
    #     Computes the log probability of actions under the Low-Rank MVN distribution.
    #     NOTE: This is a complex calculation involving the log-determinant of the
    #           covariance matrix (using matrix determinant lemma) and the Mahalanobis
    #           distance. It also requires the tanh correction if apply_tanh=True.
    #           Implementation is deferred.
    #     """
    #     act_logits = jnp.reshape(act_logits, (act_logits.shape[0],
    #                                             act_logits.shape[1],
    #                                             1,
    #                                             act_logits.shape[-1]))  
    #     # N, S, _, params = act_logits.shape

    #    #s=2, a=3, k=4
    #    #[N,S,1, s*a] [8,256,1,6]
    #     mu = act_logits[..., :self.flat_action_dim]
        

    #     #[N,S,1, s*a] [8,256,1,6]
    #     log_std = act_logits[..., self.flat_action_dim : 2 * self.flat_action_dim]
        
    #     #[N,S,1, s*a*k] [8,256,1,24]
    #     V_flat = act_logits[..., 2 * self.flat_action_dim :]
        
    #     print("V_flat shape", V_flat.shape, "act_logits shape", act_logits.shape, "mu shape", mu.shape, "log_std shape", log_std.shape)
        
    #     return actions
    #     # raise NotImplementedError("Log probability for Low-Rank MVN is not implemented yet.")
    
    def _single_log_prob(self, flat_action, mu, log_std, V):
        """
        Calculates log_prob for a single flattened action vector [d].
        Handles tanh correction internally if self.apply_tanh is True.

        Args:
            flat_action: The flattened action vector [d]. If apply_tanh=True,
                         this should be the *final* (tanh-squashed) action.
            mu: Mean vector [d].
            log_std: Log standard deviation vector for diagonal D [d].
            V: Low-rank factor matrix [d, k].

        Returns:
            The scalar log probability density.
        """
        # --- Tanh Correction Handling ---
      
        # Inverse tanh transformation: action_pre_tanh = atanh(action)
        # Clipping to avoid instabilities at +/- 1
        clipped_action = jnp.clip(flat_action, -1.0 + self.epsilon, 1.0 - self.epsilon)
        # Calculate the pre-tanh value 'x' which is assumed to be MVN distributed
        x = jnp.arctanh(clipped_action)
        # Calculate the log_prob correction term: sum(log(1 - action^2))
        # This needs to be subtracted from the MVN log_prob of 'x'
        tanh_log_det_jacobian = jnp.sum(jnp.log(1.0 - jnp.square(clipped_action) + self.epsilon))
    

        # --- MVN Calculation for 'x' ---
        y = x - mu # Deviation from mean: x - mu. Shape [d]

        # --- Setup terms for Woodbury/Determinant Lemma ---
        # D = diag(exp(2*log_std)) -> D^{-1/2} = exp(-log_std)
        D_inv_sqrt = jnp.exp(-log_std)        # [d] = 1 / std_dev
        D_inv = jnp.square(D_inv_sqrt)        # [d] = 1 / variance

        # --- Log Determinant Calculation ---
        # log det(Sigma) = log det(D) + log det(I_k + V^T * D^{-1} * V)
        # log det(D) = sum(log(variances)) = sum(2 * log_std)
        log_det_D = jnp.sum(2 * log_std)

        # Efficient calculation of M = V^T * D^{-1} * V
        # Scale columns of V by D^{-1/2} element-wise
        V_scaled = D_inv_sqrt[:, None] * V   # Shape [d, k]
        # Compute M = V_scaled^T @ V_scaled
        M = V_scaled.T @ V_scaled            # Shape [k, k]
        # Compute the inner matrix for determinant and solve: I + M
        inner_matrix = jnp.eye(self.rank_k) + M # Shape [k, k]

        # Calculate log det(I + M) using slogdet for stability
        sign, log_det_inner = jnp.linalg.slogdet(inner_matrix)
        # Ensure matrix is positive definite (sign should be 1, logdet finite)
        # If not, log_prob is -inf. Add small jitter for robustness if needed.
        log_det_inner = jnp.where(sign > 0, log_det_inner, -jnp.inf)

        log_det_Sigma = log_det_D + log_det_inner # Total log determinant

        # --- Mahalanobis Distance Calculation ---
        # Mahalanobis^2 = y^T * Sigma^{-1} * y
        # Efficient calculation using Woodbury identity results:
        # Let z = D^{-1/2} * y. Mahalanobis^2 = z^T * z - (V_scaled^T @ z)^T @ solve(I+M, V_scaled^T @ z)

        z = D_inv_sqrt * y       # Shape [d]
        V_scaled_T_z = V_scaled.T @ z # Shape [k] = V^T * D^{-1/2} * y

        # Solve (I + M) * solved_vec = V_scaled_T_z for solved_vec
        # Use Cholesky solve for stability and efficiency with SPD matrices
        try:
            # Attempt Cholesky factorization L*L^T = inner_matrix
            chol_inner = jax.scipy.linalg.cholesky(inner_matrix, lower=True)
            # Solve L*tmp = V_scaled_T_z
            tmp = jax.scipy.linalg.solve_triangular(chol_inner, V_scaled_T_z, lower=True)
            # Solve L^T*solved_vec = tmp
            solved_vec = jax.scipy.linalg.solve_triangular(chol_inner.T, tmp, lower=False)
        except jnp.linalg.LinAlgError:
             # Fallback if Cholesky fails (e.g., numerical issues)
             # Use least squares or regularized solve. Using lstsq here.
             # jax.debug.print("Warning: Cholesky failed in log_prob, using lstsq.")
             solved_vec = jax.numpy.linalg.lstsq(inner_matrix, V_scaled_T_z, rcond=None)[0]

        # Calculate Mahalanobis distance squared
        mahalanobis_sq = jnp.dot(z, z) - jnp.dot(V_scaled_T_z, solved_vec)
        # Ensure non-negativity due to potential numerical errors
        mahalanobis_sq = jnp.maximum(0.0, mahalanobis_sq)

        # --- Combine for Log Probability of 'x' ---
        # log p(x) = -0.5 * [ Mahalanobis^2 + log det(Sigma) + d * log(2*pi) ]
        log_prob_mvn = -0.5 * (mahalanobis_sq + log_det_Sigma + self.flat_action_dim * jnp.log(2 * jnp.pi))

        # --- Apply Tanh Correction ---
        # log p(action) = log p(x) - sum(log(1 - action^2))
        # where action = tanh(x)
        log_prob_final = log_prob_mvn - tanh_log_det_jacobian

        return log_prob_final

    

    def gaussian_log_prob(self, actions, act_logits):
        """
        Computes the log probability of actions under the Low-Rank MVN distribution
        for batches of trajectories.

        Args:
            actions: The actions taken. Expected shape [N, S, s, a].
                     These are the final actions (potentially tanh-squashed).
            act_logits: Raw output tensor from the policy network corresponding to the
                        states where actions were taken. Expected shape [N, S, expected_logit_size]
                        or [N, S, 1, expected_logit_size] based on user example.

        Returns:
            Log probabilities for each step. Shape [N, S].
        """
        # --- Input Shape Handling ---
        # User example showed act_logits might have an extra dim of size 1
        if act_logits.ndim == 4 and act_logits.shape[2] == 1:
             act_logits = jnp.squeeze(act_logits, axis=2) # Remove the singleton dimension
             # Expected shape [N, S, expected_logit_size]

        # Verify final shape
        if act_logits.ndim != 3 or act_logits.shape[-1] != self.expected_logit_size:
             raise ValueError(f"Unexpected act_logits shape after potential squeeze. "
                              f"Expected [N, S, {self.expected_logit_size}], got {act_logits.shape}")

        # Get dimensions N (num envs), S (num steps)
        N = actions.shape[0]
        S = actions.shape[1]

        # --- 1. Reshape actions and Parse act_logits ---
        # Reshape actions from [N, S, s, a] to [N, S, d]
        actions_flat = jnp.reshape(actions, (N, S, self.flat_action_dim))

        # Parse act_logits [N, S, expected_logit_size] into parameters
        mu = act_logits[..., :self.flat_action_dim] # Shape [N, S, d]
        log_std = act_logits[..., self.flat_action_dim : 2 * self.flat_action_dim] # Shape [N, S, d]
        V_flat = act_logits[..., 2 * self.flat_action_dim :] # Shape [N, S, d*k]
        V = jnp.reshape(V_flat, (N, S, self.flat_action_dim, self.rank_k)) # Shape [N, S, d, k]

        # --- 2. Vectorize the single log_prob calculation ---
        # Use jax.vmap to apply _single_log_prob across N and S dimensions.
        # Map over the first two axes (N, S) for all inputs.
        log_prob_calculator_vmapped = jax.vmap( # Map over N dimension (axis 0)
            jax.vmap( # Map over S dimension (axis 0 of the inner slice)
                self._single_log_prob,
                in_axes=(0, 0, 0, 0) # Map over axis 0 (S dim) for args sliced by N
            ),
            in_axes=(0, 0, 0, 0) # Map over axis 0 (N dim) for all args
        )

        # Apply the vmapped function
        log_probs = log_prob_calculator_vmapped(actions_flat, mu, log_std, V)
        
        # valid_mask = (actions[..., 0] != -2)
        # log_prob = jnp.where(valid_mask, log_prob, 0.0)
        # log_prob = jnp.sum(log_prob, axis=-1)  # shape (N, T)

        # Resulting shape should be [N, S]
        return log_probs
    
    def _single_log_det_sigma(self, log_std, V):
        """ Calculates log determinant of Sigma = D + V*V^T for a single instance """
        # log det(Sigma) = log det(D) + log det(I_k + V^T * D^{-1} * V)
        log_det_D = jnp.sum(2 * log_std) # sum log(variances)

        # Efficient calculation of M = V^T * D^{-1} * V
        D_inv_sqrt = jnp.exp(-log_std)        # [d] = 1 / std_dev
        V_scaled = D_inv_sqrt[:, None] * V   # Scale columns of V. Shape [d, k]
        M = V_scaled.T @ V_scaled            # Shape [k, k]
        inner_matrix = jnp.eye(self.rank_k) + M # Shape [k, k]

        # Calculate log det(I + M) using slogdet for stability
        sign, log_det_inner = jnp.linalg.slogdet(inner_matrix)
        log_det_inner = jnp.where(sign > 0, log_det_inner, -jnp.inf) # Handle non-PD cases

        return log_det_D + log_det_inner # Total log determinant

   
        
    # 1. Add a helper function for pre-tanh sampling (for a single instance)
    def _sample_pre_tanh(self, mu, log_std, V, key):
        """ Samples a single pre-tanh action vector x ~ MVN(mu, D+VV^T) """
        key_diag, key_rank = jax.random.split(key)
        z_diag = jax.random.normal(key_diag, shape=(self.flat_action_dim,)) # [d]
        z_rank = jax.random.normal(key_rank, shape=(self.rank_k,))          # [k]

        sqrt_D_diag = jnp.exp(log_std) # [d]
        diag_contribution = sqrt_D_diag * z_diag  # [d]
        rank_contribution = jnp.einsum('dk,k->d', V, z_rank) # [d] = V @ z_rank
        action_flat = mu + diag_contribution + rank_contribution # [d]
        return action_flat # This is the pre-tanh sample 'x'

    # 2. Add a helper function for the MC estimation of the expected log-det-Jacobian
    def _mc_expected_log_det_jacobian(self, mu, log_std, V, key, num_samples):
        """ Calculates E[ sum log(1 - tanh^2(X)) ] via Monte Carlo """
        # Generate unique keys for each sample
        sample_keys = jax.random.split(key, num_samples)

        # Vmap the pre-tanh sampling over the keys
        # sample_pre_tanh_vmapped = jax.vmap(self._sample_pre_tanh, in_axes=(None, None, None, 0)) # JAX < 0.4.14 syntax
        sample_pre_tanh_vmapped = jax.vmap(partial(self._sample_pre_tanh, mu, log_std, V), in_axes=(0)) # JAX >= 0.4.14 syntax
        
        # Generate samples
        x_samples = sample_pre_tanh_vmapped(sample_keys) # Shape [num_samples, d]

        # Calculate tanh(x) for each sample
        tanh_x_samples = jnp.tanh(x_samples) # Shape [num_samples, d]

        # Calculate log(1 - tanh^2(x)) for each element and sum over dimension d
        # log_det_jacobian_samples = jnp.sum(jnp.log(1.0 - jnp.square(tanh_x_samples) + self.epsilon), axis=-1) # Shape [num_samples]
        
        # Numerically stabler version: sum( -2 * log(cosh(x)) )
        log_cosh_x = jnp.abs(x_samples) + jnp.log1p(jnp.exp(-2.0 * jnp.abs(x_samples))) - jnp.log(2.0)
        log_det_jacobian_samples = -2.0 * jnp.sum(log_cosh_x, axis=-1) # Shape [num_samples]


        # Return the mean over the samples
        return jnp.mean(log_det_jacobian_samples)


    # 3. Modify the entropy function
    def entropy(self, act_logits, mask, key, num_entropy_samples=100):
        """
        Computes the differential entropy for batches of trajectories.

        If num_entropy_samples <= 0, computes the entropy of the base Low-Rank
        MVN distribution (H(X) where action=tanh(X)). This is standard practice.

        If num_entropy_samples > 0, approximates the entropy of the squashed
        distribution H(tanh(X)) using Monte Carlo estimation for the Jacobian term.
        H(tanh(X)) approx H(X) - E[ sum log(1 - tanh^2(X_i)) ]
        WARNING: This is computationally expensive and introduces noise.

        Args:
            act_logits: Raw output tensor from the policy network. Expected shape
                        [N, S, expected_logit_size] or [N, S, 1, ...].
            key: JAX random key. Required if num_entropy_samples > 0.
            num_entropy_samples: Number of MC samples for Jacobian term estimation.

        Returns:
            Entropy for each step. Shape [N, S].
        """
        # --- Input Shape Handling ---
        if act_logits.ndim == 4 and act_logits.shape[2] == 1:
            act_logits = jnp.squeeze(act_logits, axis=2)
        if act_logits.ndim != 3 or act_logits.shape[-1] != self.expected_logit_size:
            raise ValueError(f"Unexpected act_logits shape. Expected [N, S, {self.expected_logit_size}], got {act_logits.shape}")

        N, S = act_logits.shape[0], act_logits.shape[1]

        # --- Parse act_logits ---
        log_std = act_logits[..., self.flat_action_dim : 2 * self.flat_action_dim] # [N, S, d]
        V_flat = act_logits[..., 2 * self.flat_action_dim :] # [N, S, d*k]
        V = jnp.reshape(V_flat, (N, S, self.flat_action_dim, self.rank_k)) # [N, S, d, k]

        # --- Calculate Base Entropy H(X) ---
        # Vectorize the log determinant calculation
        log_det_sigma_calculator_vmapped = jax.vmap( # Map over N
            jax.vmap( # Map over S
                self._single_log_det_sigma, in_axes=(0, 0) # Map over S-dim slices
            ), in_axes=(0, 0) # Map over N-dim slices
        )
        log_det_sigmas = log_det_sigma_calculator_vmapped(log_std, V) # Shape [N, S]
        # H(X) = 0.5 * log det(Sigma) + 0.5 * d * (1 + log(2*pi))
        base_entropies = 0.5 * log_det_sigmas + 0.5 * self.flat_action_dim * (1.0 + jnp.log(2.0 * jnp.pi)) # Shape [N, S]

        # --- Optional: Calculate MC Correction Term ---
       

        # Parse mu (needed for sampling)
        mu = act_logits[..., :self.flat_action_dim] # [N, S, d]

        # Generate keys for each (N, S) step and the MC samples within
        # Create a grid of keys for N, S
        step_keys = jax.random.split(key, N * S).reshape(N, S, -1) # Shape [N, S, 2]

        # Vmap the MC expectation calculation over N and S
        mc_expectation_vmapped = jax.vmap( # Map over N
            jax.vmap( # Map over S
                self._mc_expected_log_det_jacobian,
                # Pass mu, log_std, V slices, step_key, and static num_samples
                in_axes=(0, 0, 0, 0, None)
            ), in_axes=(0, 0, 0, 0, None) # Map over N-dim slices
        )

        # Calculate expected log_det_jacobian term E[ sum log(1 - tanh^2(X_i)) ]
        expected_log_det_jacobians = mc_expectation_vmapped(
            mu, log_std, V, step_keys, num_entropy_samples
        ) # Shape [N, S]

        # Corrected Entropy H(Y) = H(X) - E[ log |det J| ]
        entropies = base_entropies + expected_log_det_jacobians
        # entropies = expected_log_det_jacobians
        print("ewtbsg", entropies.shape)
        jax.debug.print("base {} correction {}", base_entropies[0, 0], expected_log_det_jacobians[0,0])
       

        return entropies