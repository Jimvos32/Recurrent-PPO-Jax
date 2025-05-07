# Required imports (ensure flax is installed: pip install flax)
from functools import partial
import jax
from jax import config, jit, value_and_grad
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import gpjax as gpx
from scipy.optimize import minimize
# Use flax for JAX-friendly dataclasses
# pip install flax
from flax import struct
import warnings
import optax as ox


# Configure JAX
config.update("jax_enable_x64", True)

# --- Define State and Static Parameters ---

@struct.dataclass
class BOState:
    """Holds the mutable state of the Bayesian Optimizer."""
    X: jnp.ndarray
    Y: jnp.ndarray
    key: jr.PRNGKey
    # Store the optimized posterior model directly.
    # Note: GPJax models are often pytrees, suitable for JAX.
    # Use 'Any' type hint as the exact posterior type might vary slightly.
    # We might need to handle the 'None' case during initialization carefully.
    posterior: object # Could be gpx.gps.Posterior or None initially
    dataset: gpx.Dataset | None # Keep track of the dataset object as well

@struct.dataclass
class BOStaticParams:
    """Holds the static configuration of the Bayesian Optimizer."""
    input_dim: int
    min_bounds: jnp.ndarray
    max_bounds: jnp.ndarray
    prior: gpx.gps.Prior
    kappa: float
    # Add likelihood constructor args if needed, e.g., fixed noise variance
    # likelihood_params: dict = struct.field(default_factory=dict)

# --- Core Pure Functions ---

def _build_likelihood(n_datapoints, **likelihood_params):
    """Builds the Gaussian likelihood."""
    # In this version, likelihood is simple, but could be configured via likelihood_params
    return gpx.likelihoods.Gaussian(num_datapoints=n_datapoints)



# Note: GP fitting with SciPy optimizer inside isn't directly JIT-compilable.
# `gpx.fit` often uses non-JAX optimizers internally.
# We JIT the MLL calculation, but the optimization loop itself runs outside JIT.
def fit_gp_model(state: BOState, static_params: BOStaticParams) -> object:
    """
    Fits the GP hyperparameters. Returns the optimized posterior object.
    This function is NOT pure JAX due to potential scipy usage in gpx.fit,
    but it encapsulates the fitting logic.
    """
    if state.dataset is None or state.dataset.n == 0:
        warnings.warn("No data available to fit the model.", stacklevel=2)
        return None # Or return the prior? Depends on desired behavior

    likelihood = _build_likelihood(state.dataset.n)
    current_posterior_unfitted = static_params.prior * likelihood

    @partial(jax.jit, static_argnums=(0))
    def negative_mll(posterior, dataset):
        mll = gpx.objectives.conjugate_mll(posterior, dataset)
        return -mll
    # --- END CORRECTION ---

    try:
        # Pass the JIT-compiled negative_mll function to the optimizer.
        # `gpx.fit` expects a function that takes (model, data) and returns scalar loss.
        #this is how it is called in the gpjax library demo
        # optimizer must be a optax GradientTransformation but I would like to keep it plain and let the library do what it normally does
        
        optimised_posterior, history = gpx.fit_scipy(
            model=current_posterior_unfitted,
            objective=negative_mll, # Pass the JITted function
            train_data=state.dataset,
            max_iters=100,
            # optim=optimizer,
        )

        # Calculate the final *negative* MLL value using the optimized posterior
        # final_nll_val = negative_mll(optimised_posterior, state.dataset)

        return optimised_posterior

    except Exception as e:
        print(f"Error during GP optimization: {e}")
        warnings.warn(f"Error during GP optimization: {e}. Returning unfitted posterior.", stacklevel=2)
        return current_posterior_unfitted


@partial(jax.jit, static_argnums=(1,))
def calculate_ucb(x_candidate: jnp.ndarray,
                  posterior: gpx.gps.AbstractPosterior, # Static arg index 1
                  X_train: jnp.ndarray,             # Dynamic JAX array
                  y_train: jnp.ndarray,             # Dynamic JAX array
                  kappa: float) -> jnp.ndarray:
    """
    Calculates the Upper Confidence Bound (UCB). JIT-compilable.
    Accepts X_train, y_train directly to avoid passing unhashable Dataset.
    Only 'posterior' needs to be static.
    Assumes maximization of the objective function.
    """
    x_candidate_2d = jnp.atleast_2d(x_candidate)
    
    # X_train = jnp.zeros((X_train.shape[0], X_train.shape[1]), dtype=jnp.float64)
    # y_train = jnp.zeros((y_train.shape[0], y_train.shape[1]), dtype=jnp.float64)
    # Create a temporary Dataset inside JIT scope if posterior.predict needs it.
    # This is generally fine for JIT as Dataset creation is traceable.
    train_data = gpx.Dataset(X=X_train, y=y_train)

    # Predict latent function distribution using the static posterior and dynamic data
    latent_dist = posterior.predict(x_candidate_2d, train_data=train_data)
    
    predictive_dist  = posterior.likelihood(latent_dist) # Ensure likelihood is called for correct prediction
    
    mean = predictive_dist.mean
    std_dev = jnp.sqrt(predictive_dist.variance + 1e-6) # Variance from likelihood   

    # mean = latent_dist.mean()
    # variance = latent_dist.variance()
    # std_dev = jnp.sqrt(jnp.maximum(variance, 1e-12)) # Epsilon for stability
    
    # jax.debug.print("UCB mean: {} std_dev: {} canidate {}", mean, std_dev, x_candidate_2d)

    ucb_value = mean + kappa * std_dev
    return ucb_value.squeeze() # Return scalar


# This function orchestrates the acquisition optimization, likely calling non-JAX code (scipy)
def optimize_acquisition(state: BOState, static_params: BOStaticParams, n_restarts: int) -> tuple[jnp.ndarray, float, jr.PRNGKey]:
    """
    Finds the point maximizing UCB using scipy.optimize.minimize.
    Extracts X, y from state.dataset to pass to calculate_ucb.
    obj_fn remains NOT JITted.
    Returns the best point, its UCB value, and the updated key.
    """
    if state.posterior is None or state.dataset is None:
        raise ValueError("GP model must be fitted before optimizing acquisition.")
    # Ensure dataset actually has data before accessing .X, .Y
    if state.dataset.n == 0:
        raise ValueError("Dataset is empty, cannot optimize acquisition.")


    key, subkey = jr.split(state.key)
    bounds = list(zip(static_params.min_bounds, static_params.max_bounds))
    best_acq_value = -jnp.inf
    best_x = None

    # Capture posterior (static), extract X/y (dynamic), capture kappa (dynamic)
    posterior_for_opt = state.posterior
    X_train_for_opt = state.dataset.X # Extract JAX array X
    y_train_for_opt = state.dataset.y # Extract JAX array y
    kappa_for_opt = static_params.kappa

    # Define the function to minimize (negated UCB) - NOT JITTED
    def obj_fn(x):
         # Call the JITted calculate_ucb, passing X_train and y_train directly
         return -calculate_ucb(x,
                                posterior_for_opt,
                                X_train_for_opt, # Pass array
                                y_train_for_opt, # Pass array
                                kappa_for_opt)

    # print("min", bounds, static_params.min_bounds, static_params.max_bounds)
    random_starts = jr.uniform(subkey, (n_restarts, static_params.input_dim),
                               minval=static_params.min_bounds,
                               maxval=static_params.max_bounds)

    for start_point in random_starts:
        # Pass the non-JITted obj_fn wrapper to minimize.
        res = minimize(fun=obj_fn,
                       x0=start_point,
                       method='L-BFGS-B',
                       bounds=bounds)
        
        if res.success:
            if -res.fun > best_acq_value:
                best_acq_value = -res.fun
                best_x = res.x
                # jax.debug.print("Best point found in this restart: {} {}", best_x, best_acq_value)

    if best_x is None:
        warnings.warn("Acquisition optimization failed. Returning random point.", stacklevel=2)
        key, subkey = jr.split(key)
        best_x = jr.uniform(subkey, (static_params.input_dim,), minval=static_params.min_bounds, maxval=static_params.max_bounds)
        best_acq_value = -obj_fn(best_x)
    else:
        best_x = jnp.array(best_x)

    # Return the best point, its acquisition value, and the updated key state
    return best_x.reshape(-1), best_acq_value, key

def update_step(state: BOState, static_params: BOStaticParams, x_new: jnp.ndarray, y_new: jnp.ndarray) -> BOState:
    """
    Pure function to update the state with new data and refit the model.
    """
    x_new = jnp.atleast_2d(x_new)
    y_new = jnp.atleast_2d(y_new)

    # Append new data
    
    
    new_X = jnp.concatenate([state.X, x_new], axis=0)
    new_Y = jnp.concatenate([state.Y, y_new], axis=0)

    # Update GPJax Dataset
    new_dataset = gpx.Dataset(X=new_X, y=new_Y)

    # Create a temporary state with new data for fitting
    state_for_fitting = state.replace(X=new_X, Y=new_Y, dataset=new_dataset)

    # Fit the model using the updated data
    # This step is not JIT-compilable if gpx.fit uses non-jax optimizers
    jax.debug.print("Fitting GP model with new data... {} {}", new_X.shape[0], new_X.shape[1])
    new_posterior = fit_gp_model(state_for_fitting, static_params)

    # Return the completely new state
    return state.replace(
        X=new_X,
        Y=new_Y,
        posterior=new_posterior,
        dataset=new_dataset
    )

# def suggest_step(state: BOState, static_params: BOStaticParams, n_restarts: int = 10) -> tuple[jnp.ndarray, BOState]:
#     """
#     Pure function to suggest the next point based on the current state.
#     Returns the suggestion and the updated state (key).
#     """
#     key = state.key # Get current key

#     # Handle cases where suggestion is not possible/meaningful
#     if state.X.shape[0] < 1 or state.posterior is None or state.dataset is None:
#          key, subkey = jr.split(key)
#          suggestion = jr.uniform(subkey, (static_params.input_dim,), minval=static_params.min_bounds, maxval=static_params.max_bounds)
#          # Update state only with the new key
#          new_state = state.replace(key=key)
#          return suggestion, new_state

#     # Optimize acquisition function
#     # This step is not JIT-compilable if using scipy.optimize
#     suggestion, best_acq_value, updated_key = optimize_acquisition(state, static_params, n_restarts)

#     # Create the new state with the updated key
#     new_state = state.replace(key=updated_key)

#     return suggestion, new_state

def suggest_batch_step(state: BOState, static_params: BOStaticParams, batch_size: int, n_restarts: int = 10) -> tuple[jnp.ndarray, BOState]:
    """
    Pure function to suggest a BATCH of points using sequential greedy UCB.
    Returns the batch of suggestions and the updated state (key).
    """
    current_key = state.key # Start with the key from the input state

    # Handle cases where suggestion is not possible/meaningful
    if state.X.shape[0] < 1 or state.posterior is None or state.dataset is None:
         current_key, subkey = jr.split(current_key)
         suggestions = jr.uniform(subkey, (batch_size, static_params.input_dim,),
                                  minval=static_params.min_bounds, maxval=static_params.max_bounds)
         # Update state only with the new key
         new_state = state.replace(key=current_key)
         return suggestions, new_state

    # --- Sequential Greedy Batch Selection ---
    batch_points = []
    # Start with the real observed data
    fantasy_X = state.X
    fantasy_Y = state.Y
    # Use the single posterior fitted on the real data for all fantasy steps
    fixed_posterior = state.posterior

    for i in range(batch_size):
        # Create dataset based on current fantasy data
        current_fantasy_dataset = gpx.Dataset(X=fantasy_X, y=fantasy_Y)

        # Create a temporary state for optimize_acquisition
        # Use the *fixed* posterior but *updated* fantasy dataset and current key
        temp_state_for_opt = BOState( # Construct directly or use .replace if base state is suitable
            X=fantasy_X,
            Y=fantasy_Y,
            posterior=fixed_posterior,
            dataset=current_fantasy_dataset,
            key=current_key # Pass the current key to the optimizer
        )

        # Find the best point given current fantasy data
        x_next, _, updated_key = optimize_acquisition(temp_state_for_opt, static_params, n_restarts)
        current_key = updated_key # Use the key returned by optimize_acquisition for the next step

        # Store the suggested point
        # Ensure it has the correct shape (input_dim,) before appending? optimize_acquisition returns (dim,)
        batch_points.append(x_next)

        # If not the last point, create fantasy observation for the next iteration
        if i < batch_size - 1:
            # Use the fixed posterior, conditioned on the current fantasy data, to predict the mean
            x_next_2d = jnp.atleast_2d(x_next) # Ensure shape (1, dim)
            latent_dist = fixed_posterior.predict(x_next_2d, train_data=current_fantasy_dataset)
            predictive_dist = fixed_posterior.likelihood(latent_dist) # Ensure likelihood is called for correct prediction
            y_fantasy = predictive_dist.mean # Get the mean prediction
            y_fantasy = jnp.atleast_2d(y_fantasy) # Ensure shape (1, 1)

            # Augment fantasy data for the *next* iteration
            fantasy_X = jnp.concatenate([fantasy_X, x_next_2d], axis=0)
            fantasy_Y = jnp.concatenate([fantasy_Y, y_fantasy], axis=0)
         
    # Convert list of points to a JAX array -> shape (batch_size, input_dim)
    final_batch = jnp.stack(batch_points, axis=0)

    # Create the final state to return: Use the original state's X, Y, posterior, dataset
    # but update the key to the latest one used/returned by the process.
    final_state = state.replace(key=current_key)

    return final_batch, final_state


# --- Class Wrapper ---

class BayesianOptimizer:
    """
    Manages the Bayesian Optimization process using pure step functions.

    Args:
        search_space_bounds (tuple): (min_bounds, max_bounds) for each input dimension.
        kernel (gpx.kernels.Kernel, optional): GPJax kernel. Defaults to RBF.
        mean_function (gpx.mean_functions.MeanFunction, optional): GPJax mean function. Defaults to Zero.
        acquisition_kappa (float, optional): Kappa for UCB. Defaults to 1.96.
        key (jr.PRNGKey, optional): JAX random key. Defaults to jr.key(123).
    """
    def __init__(self,
                 search_space_bounds: tuple[jnp.ndarray, jnp.ndarray],
                 kernel: gpx.kernels.AbstractKernel = None,
                 mean_function: gpx.mean_functions.AbstractMeanFunction = None,
                 acquisition_kappa: float = 1.96,
                 key: jr.PRNGKey = jr.key(123)
                 ):

        min_bounds, max_bounds = search_space_bounds
        if min_bounds.shape != max_bounds.shape:
             raise ValueError("min_bounds and max_bounds must have the same shape.")
        input_dim = min_bounds.shape[0]

        # Setup static parameters
        print("dfg", list(range(input_dim)))
        
        # _kernel = kernel if kernel is not None else gpx.kernels.RBF(active_dims=list(range(input_dim)))
        _kernel = kernel if kernel is not None else gpx.kernels.RBF()
        print("dfg", _kernel.active_dims)
        _mean_function = mean_function if mean_function is not None else gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(mean_function=_mean_function, kernel=_kernel)

        self.static_params = BOStaticParams(
            input_dim=input_dim,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            prior=prior,
            kappa=acquisition_kappa
        )

        # Initialize state
        initial_X = jnp.empty((0, input_dim), dtype=jnp.float64)
        initial_Y = jnp.empty((0, 1), dtype=jnp.float64)
        self.state = BOState(
            X=initial_X,
            Y=initial_Y,
            key=key,
            posterior=None, # No posterior initially
            dataset=None    # No dataset initially
        )

       


    def update(self, x_new: jnp.ndarray, y_new: jnp.ndarray):
        """
        Updates the optimizer state with new observed data points by calling the pure update_step function.
        """
        # Ensure shapes are okay before passing to step function
        x_new_arr = jnp.atleast_2d(x_new)
        y_new_arr = jnp.atleast_2d(y_new)
        # if x_new_arr.shape[1] != self.static_params.input_dim:
        #      raise ValueError(f"Input x_new has wrong dimension {x_new_arr.shape[1]}, expected {self.static_params.input_dim}")
        # if y_new_arr.shape[1] != 1:
        #      raise ValueError(f"Input y_new must have shape (n, 1), got {y_new_arr.shape}")
        # if x_new_arr.shape[0] != y_new_arr.shape[0]:
        #      raise ValueError("Number of points in x_new and y_new must match.")

        # Call the pure function, get the new state back
        self.state = update_step(self.state, self.static_params, x_new_arr, y_new_arr)

    def suggest(self, batch_size: int = 1, n_restarts: int = 10) -> jnp.ndarray:
        """
        Suggests the next batch of points to sample.

        Args:
            batch_size (int): The number of points to suggest in the batch. Defaults to 1.
            n_restarts (int): Number of random restarts for acquisition function optimization
                              within each step of batch generation.

        Returns:
            jnp.ndarray: The suggested batch of input points, shape (batch_size, input_dim).
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        # Call the batch suggestion step function
        suggestions, new_state = suggest_batch_step(
            self.state,
            self.static_params,
            batch_size,
            n_restarts
        )

        # Update the optimizer's state (mainly the key)
        self.state = new_state

        # Return the batch of suggestions
        # If batch_size was 1, shape is (1, dim), otherwise (batch_size, dim)
        return suggestions

    # --- Helper properties to access state if needed ---
    @property
    def X(self) -> jnp.ndarray:
        return self.state.X

    @property
    def Y(self) -> jnp.ndarray:
        return self.state.Y

    @property
    def posterior(self) -> object:
        return self.state.posterior


# # --- Example Usage (Identical to previous, but uses the new structure) ---
# if __name__ == '__main__':

#     # Define the 'environment' function (the function to optimize)
#     def objective_function(x: jnp.ndarray) -> jnp.ndarray:
#         x_col = x.reshape(-1, 1)
#         val = jnp.sin(4 * x_col) + jnp.cos(2 * x_col)
#         key = jr.PRNGKey(int(jnp.sum(x)*1000 + jnp.prod(x)*500)) # Slightly better key gen
#         noise = 0.1
#         noisy_val = val + jr.normal(key, shape=val.shape) * noise
#         return noisy_val.reshape(-1, 1)

#     # --- BO Setup ---
#     search_bounds = (jnp.array([-3.0]), jnp.array([3.0])) # 1D search space
#     bo_optimizer = BayesianOptimizer(search_space_bounds=search_bounds, key=jr.PRNGKey(42))

#     # --- Initial Data ---
#     initial_X = jnp.array([[0.0]])
#     initial_Y = objective_function(initial_X)
#     bo_optimizer.update(initial_X, initial_Y) # Update uses the new structure

#     # --- The Optimization Loop ---
#     n_iterations = 15

#     print("\n--- Starting Bayesian Optimization Loop ---")
#     for i in range(n_iterations):
#         print(f"\n--- Iteration {i+1}/{n_iterations} ---")
#         # 1. Get suggestion from BO (suggest uses the new structure)
#         suggested_x = bo_optimizer.suggest(n_restarts=10)

#         suggested_x_reshaped = suggested_x.reshape(1, -1) # Ensure shape (1, input_dim)

#         # 2. Query the 'environment'
#         observed_y = objective_function(suggested_x_reshaped)

#         print(f"Sampled point: {suggested_x.flatten()}, Observed value: {observed_y.item():.4f}")

#         # 3. Update the BO model (update uses the new structure)
#         bo_optimizer.update(suggested_x_reshaped, observed_y)

#     # --- Results ---
#     print("\n--- Optimization Finished ---")
#     # Access data via properties that read from the state
#     print(f"Observed data points (X): \n{bo_optimizer.X}")
#     print(f"Observed data points (Y): \n{bo_optimizer.Y}")

#     best_idx = jnp.argmax(bo_optimizer.Y)
#     best_x_found = bo_optimizer.X[best_idx]
#     best_y_found = bo_optimizer.Y[best_idx]

#     print(f"\nBest point found: X = {best_x_found}, Y = {best_y_found.item():.4f}")

#     # --- Optional: Plotting (accesses posterior via property) ---
#     if bo_optimizer.static_params.input_dim == 1 and bo_optimizer.posterior is not None:
#         import matplotlib.pyplot as plt
#         import matplotlib as mpl
#         cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

#         xtest = jnp.linspace(bo_optimizer.static_params.min_bounds[0], bo_optimizer.static_params.max_bounds[0], 200).reshape(-1, 1)

#         # Predict using the final posterior and dataset from the state
#         final_posterior = bo_optimizer.posterior
#         final_dataset = bo_optimizer.state.dataset

#         latent_dist = final_posterior.predict(xtest, train_data=final_dataset)
#         predictive_dist = final_posterior.likelihood(latent_dist)

#         pred_mean = predictive_dist.mean()
#         pred_std = jnp.sqrt(predictive_dist.variance())

#         f_true = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
#         ytest_true = f_true(xtest)

#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.plot(bo_optimizer.X, bo_optimizer.Y, "x", label="Observations", color=cols[0], markersize=8)
#         ax.plot(xtest, pred_mean, label="Predictive Mean", color=cols[1])
#         ax.fill_between(
#              xtest.flatten(),
#              pred_mean.flatten() - 1.96 * pred_std.flatten(),
#              pred_mean.flatten() + 1.96 * pred_std.flatten(),
#              alpha=0.2, color=cols[1], label="95% Confidence Interval"
#          )
#         ax.plot(xtest, ytest_true, 'k--', label="True Function (noiseless)")
#         ax.scatter(best_x_found, best_y_found, color='red', s=100, zorder=5, label=f'Best Observed Point')

#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_title("Bayesian Optimization Result (Stateful Class, Pure Steps)")
#         ax.legend()
#         ax.grid(True, which='both', linestyle='--', linewidth=0.5)
#         plt.tight_layout()
#         plt.show()