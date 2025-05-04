# Required imports (ensure flax is installed: pip install flax)
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

@jit
def _conjugate_mll(posterior: gpx.gps.Posterior, dataset: gpx.Dataset):
    """JIT-compilable Marginal Log-Likelihood calculation."""
    # Using the ConjugateMLL objective requires the posterior object
    # Note: GPJax objectives might directly take model+data. Check specific version.
    # This assumes ConjugateMLL is compatible with this structure.
    # The negative=True is handled by the caller if minimizing.
    return gpx.objectives.ConjugateMLL(negative=False)(posterior, dataset)


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

    # Using the negative MLL for minimization by the optimizer
    objective = jit(gpx.objectives.ConjugateMLL(negative=True))

    print("Optimizing GP hyperparameters...")
    try:
        # Using gpx.fit which often wraps scipy or other optimizers.
        # This step is usually the boundary of JIT compilation.
        optimised_posterior, history = gpx.fit(
            model=current_posterior_unfitted,
            objective=objective,
            train_data=state.dataset,
            # optimizer=gpx.optimizers.Scipy(), # Specify if needed
            num_iters=100 # Default for scipy optimizer in older gpjax
        )
        mll_val = objective(optimised_posterior, state.dataset)
        print(f"Optimization complete. Final Neg MLL: {mll_val:.4f}")
        return optimised_posterior

    except Exception as e:
         warnings.warn(f"Error during GP optimization: {e}. Returning unfitted posterior.", stacklevel=2)
         # Fallback: Return the unfitted posterior based on current data
         return current_posterior_unfitted


@jit
def calculate_ucb(x_candidate: jnp.ndarray,
                  posterior: gpx.gps.Posterior, # Pass the fitted posterior
                  dataset: gpx.Dataset,       # Pass the corresponding dataset
                  kappa: float) -> jnp.ndarray:
    """
    Calculates the Upper Confidence Bound (UCB). JIT-compilable.
    Assumes maximization of the objective function.
    """
    # Reshape for prediction: GPJax expects (n_points, n_features)
    x_candidate_2d = jnp.atleast_2d(x_candidate)

    # Predict latent function distribution using the provided posterior and dataset
    latent_dist = posterior.predict(x_candidate_2d, train_data=dataset)

    mean = latent_dist.mean()
    variance = latent_dist.variance()
    std_dev = jnp.sqrt(jnp.maximum(variance, 1e-12)) # Epsilon for stability

    ucb_value = mean + kappa * std_dev
    return ucb_value.squeeze() # Return scalar


# This function orchestrates the acquisition optimization, likely calling non-JAX code (scipy)
def optimize_acquisition(state: BOState, static_params: BOStaticParams, n_restarts: int) -> tuple[jnp.ndarray, float, jr.PRNGKey]:
    """
    Finds the point maximizing UCB using scipy.optimize.minimize.
    Returns the best point, its UCB value, and the updated key.
    """
    if state.posterior is None or state.dataset is None:
        raise ValueError("GP model must be fitted before optimizing acquisition.")

    key, subkey = jr.split(state.key)
    bounds = list(zip(static_params.min_bounds, static_params.max_bounds))

    best_acq_value = -jnp.inf
    best_x = None

    # Define the objective for scipy.optimize.minimize (negative UCB)
    # We capture the necessary parts of the state and params for the objective call
    posterior_for_opt = state.posterior
    dataset_for_opt = state.dataset
    kappa_for_opt = static_params.kappa

    # Define the function to minimize (negated UCB)
    # This internal function `obj_fn` can be JITted for faster evaluation
    @jit
    def obj_fn(x):
         return -calculate_ucb(x, posterior_for_opt, dataset_for_opt, kappa_for_opt)

    print(f"Optimizing UCB acquisition function with {n_restarts} restarts...")
    random_starts = jr.uniform(subkey, (n_restarts, static_params.input_dim),
                               minval=static_params.min_bounds,
                               maxval=static_params.max_bounds)

    for start_point in random_starts:
        res = minimize(fun=obj_fn, # Pass the JITted evaluation function
                       x0=start_point,
                       method='L-BFGS-B',
                       bounds=bounds)

        if res.success:
             # Value of obj_fn is -UCB, so res.fun is -UCB
             # We want to maximize UCB, so we compare -res.fun
            if -res.fun > best_acq_value:
                best_acq_value = -res.fun
                best_x = res.x
        # else:
        #     print(f"Optimizer restart failed from {start_point}. Message: {res.message}")

    if best_x is None:
        warnings.warn("Acquisition optimization failed. Returning random point.", stacklevel=2)
        key, subkey = jr.split(key)
        best_x = jr.uniform(subkey, (static_params.input_dim,), minval=static_params.min_bounds, maxval=static_params.max_bounds)
        # Evaluate UCB at the random point if possible
        best_acq_value = -obj_fn(best_x) # Calculate UCB for the random point
    else:
        # Ensure best_x is a JAX array for consistency
        best_x = jnp.array(best_x)


    print(f"Suggesting point: {best_x} with UCB value: {best_acq_value:.4f}")
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
    print(f"Dataset updated. Total points: {new_dataset.n}")

    # Create a temporary state with new data for fitting
    state_for_fitting = state.replace(X=new_X, Y=new_Y, dataset=new_dataset)

    # Fit the model using the updated data
    # This step is not JIT-compilable if gpx.fit uses non-jax optimizers
    new_posterior = fit_gp_model(state_for_fitting, static_params)

    # Return the completely new state
    return state.replace(
        X=new_X,
        Y=new_Y,
        posterior=new_posterior,
        dataset=new_dataset
    )

def suggest_step(state: BOState, static_params: BOStaticParams, n_restarts: int = 10) -> tuple[jnp.ndarray, BOState]:
    """
    Pure function to suggest the next point based on the current state.
    Returns the suggestion and the updated state (key).
    """
    key = state.key # Get current key

    # Handle cases where suggestion is not possible/meaningful
    if state.X.shape[0] < 1 or state.posterior is None or state.dataset is None:
         print("Insufficient data or model not fitted. Suggesting a random point.")
         key, subkey = jr.split(key)
         suggestion = jr.uniform(subkey, (static_params.input_dim,), minval=static_params.min_bounds, maxval=static_params.max_bounds)
         # Update state only with the new key
         new_state = state.replace(key=key)
         return suggestion, new_state

    # Optimize acquisition function
    # This step is not JIT-compilable if using scipy.optimize
    suggestion, best_acq_value, updated_key = optimize_acquisition(state, static_params, n_restarts)

    # Create the new state with the updated key
    new_state = state.replace(key=updated_key)

    return suggestion, new_state


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
                 kernel: gpx.kernels.Kernel = None,
                 mean_function: gpx.mean_functions.AbstractMeanFunction = None,
                 acquisition_kappa: float = 1.96,
                 key: jr.PRNGKey = jr.key(123)
                 ):

        min_bounds, max_bounds = search_space_bounds
        if min_bounds.shape != max_bounds.shape:
             raise ValueError("min_bounds and max_bounds must have the same shape.")
        input_dim = min_bounds.shape[0]

        # Setup static parameters
        _kernel = kernel if kernel is not None else gpx.kernels.RBF(active_dims=list(range(input_dim)))
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

        print(f"Initialized BayesianOptimizer with {input_dim} input dimension(s).")
        print(f"Search space bounds: Min={min_bounds}, Max={max_bounds}")


    def update(self, x_new: jnp.ndarray, y_new: jnp.ndarray):
        """
        Updates the optimizer state with new observed data points by calling the pure update_step function.
        """
        # Ensure shapes are okay before passing to step function
        x_new_arr = jnp.atleast_2d(x_new)
        y_new_arr = jnp.atleast_2d(y_new)
        if x_new_arr.shape[1] != self.static_params.input_dim:
             raise ValueError(f"Input x_new has wrong dimension {x_new_arr.shape[1]}, expected {self.static_params.input_dim}")
        if y_new_arr.shape[1] != 1:
             raise ValueError(f"Input y_new must have shape (n, 1), got {y_new_arr.shape}")
        if x_new_arr.shape[0] != y_new_arr.shape[0]:
             raise ValueError("Number of points in x_new and y_new must match.")

        # Call the pure function, get the new state back
        self.state = update_step(self.state, self.static_params, x_new_arr, y_new_arr)

    def suggest(self, n_restarts: int = 10) -> jnp.ndarray:
        """
        Suggests the next point by calling the pure suggest_step function.
        Updates the internal state (key).
        """
        # Call the pure function
        suggestion, new_state = suggest_step(self.state, self.static_params, n_restarts)

        # Update the optimizer's state (mainly the key)
        self.state = new_state

        return suggestion # Return only the suggestion

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


# --- Example Usage (Identical to previous, but uses the new structure) ---
if __name__ == '__main__':

    # Define the 'environment' function (the function to optimize)
    def objective_function(x: jnp.ndarray) -> jnp.ndarray:
        x_col = x.reshape(-1, 1)
        val = jnp.sin(4 * x_col) + jnp.cos(2 * x_col)
        key = jr.PRNGKey(int(jnp.sum(x)*1000 + jnp.prod(x)*500)) # Slightly better key gen
        noise = 0.1
        noisy_val = val + jr.normal(key, shape=val.shape) * noise
        return noisy_val.reshape(-1, 1)

    # --- BO Setup ---
    search_bounds = (jnp.array([-3.0]), jnp.array([3.0])) # 1D search space
    bo_optimizer = BayesianOptimizer(search_space_bounds=search_bounds, key=jr.PRNGKey(42))

    # --- Initial Data ---
    initial_X = jnp.array([[0.0]])
    initial_Y = objective_function(initial_X)
    bo_optimizer.update(initial_X, initial_Y) # Update uses the new structure

    # --- The Optimization Loop ---
    n_iterations = 15

    print("\n--- Starting Bayesian Optimization Loop ---")
    for i in range(n_iterations):
        print(f"\n--- Iteration {i+1}/{n_iterations} ---")
        # 1. Get suggestion from BO (suggest uses the new structure)
        suggested_x = bo_optimizer.suggest(n_restarts=10)

        suggested_x_reshaped = suggested_x.reshape(1, -1) # Ensure shape (1, input_dim)

        # 2. Query the 'environment'
        observed_y = objective_function(suggested_x_reshaped)

        print(f"Sampled point: {suggested_x.flatten()}, Observed value: {observed_y.item():.4f}")

        # 3. Update the BO model (update uses the new structure)
        bo_optimizer.update(suggested_x_reshaped, observed_y)

    # --- Results ---
    print("\n--- Optimization Finished ---")
    # Access data via properties that read from the state
    print(f"Observed data points (X): \n{bo_optimizer.X}")
    print(f"Observed data points (Y): \n{bo_optimizer.Y}")

    best_idx = jnp.argmax(bo_optimizer.Y)
    best_x_found = bo_optimizer.X[best_idx]
    best_y_found = bo_optimizer.Y[best_idx]

    print(f"\nBest point found: X = {best_x_found}, Y = {best_y_found.item():.4f}")

    # --- Optional: Plotting (accesses posterior via property) ---
    if bo_optimizer.static_params.input_dim == 1 and bo_optimizer.posterior is not None:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

        xtest = jnp.linspace(bo_optimizer.static_params.min_bounds[0], bo_optimizer.static_params.max_bounds[0], 200).reshape(-1, 1)

        # Predict using the final posterior and dataset from the state
        final_posterior = bo_optimizer.posterior
        final_dataset = bo_optimizer.state.dataset

        latent_dist = final_posterior.predict(xtest, train_data=final_dataset)
        predictive_dist = final_posterior.likelihood(latent_dist)

        pred_mean = predictive_dist.mean()
        pred_std = jnp.sqrt(predictive_dist.variance())

        f_true = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
        ytest_true = f_true(xtest)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(bo_optimizer.X, bo_optimizer.Y, "x", label="Observations", color=cols[0], markersize=8)
        ax.plot(xtest, pred_mean, label="Predictive Mean", color=cols[1])
        ax.fill_between(
             xtest.flatten(),
             pred_mean.flatten() - 1.96 * pred_std.flatten(),
             pred_mean.flatten() + 1.96 * pred_std.flatten(),
             alpha=0.2, color=cols[1], label="95% Confidence Interval"
         )
        ax.plot(xtest, ytest_true, 'k--', label="True Function (noiseless)")
        ax.scatter(best_x_found, best_y_found, color='red', s=100, zorder=5, label=f'Best Observed Point')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Bayesian Optimization Result (Stateful Class, Pure Steps)")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()