import numpy as np
import abc
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, solve_triangular
from src.tasks.envs.function_envs.sampling_functions.base_sampler import FunctionSampler

class GPSampler(FunctionSampler):
    """ Samples a function instance from a GP prior (Matérn 5/2 kernel). """
    def __init__(self, action_dim, x_range, config=None):
        super().__init__(action_dim, x_range, config)
        # GP Hyperparameters (provide defaults, allow override via config)
        self.length_scale = float(self.config.get("length_scale", 1.0))
        self.sigma_f = float(self.config.get("sigma_f", 1.0)) # Signal variance
        self.sigma_n = float(self.config.get("sigma_n", 1e-5)) # Noise/Jitter for numerical stability

        # Grid configuration
        # Use points_per_dim OR total_points. points_per_dim is easier for scaling d.
        points_per_dim = int(self.config.get("points_per_dim", 10))
        # total_points = int(self.config.get("total_points", 1000)) # Alternative config
        if points_per_dim ** action_dim > 5000 and action_dim > 3: # Heuristic limit
             print(f"Warning: Grid size {points_per_dim}^{action_dim} is very large. Consider reducing points_per_dim.")

        self.grid_points_per_dim = points_per_dim
        self.num_grid_points = points_per_dim ** action_dim

        # State variables to be set during initialization
        self.X_grid = None
        self.y_grid_sample = None
        self._K_chol = None # Cholesky decomposition of kernel matrix
        self._K_inv_y = None # Store K_inv @ y_sample for efficient prediction

    def _matern52_kernel(self, X1, X2=None):
        """ Computes the Matérn 5/2 kernel between X1 and X2. """
        if X2 is None:
            X2 = X1
        # Pairwise distances (Euclidean)
        # Ensure inputs are 2D
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        
        # Use cdist for efficient distance calculation
        dists = cdist(X1 / self.length_scale, X2 / self.length_scale, metric='euclidean')

        # Matérn 5/2 formula
        sqrt5_dists = np.sqrt(5) * dists
        term1 = 1 + sqrt5_dists + (5.0/3.0) * dists**2
        kernel_matrix = self.sigma_f**2 * term1 * np.exp(-sqrt5_dists)

        # Add nugget for stability only on diagonal when X1 is X2
        if np.array_equal(X1, X2):
             np.fill_diagonal(kernel_matrix, kernel_matrix.diagonal() + self.sigma_n**2)

        return kernel_matrix

    def initialize(self):
        """ Samples a function from the GP prior and estimates bounds. """
        # 1. Create grid
        linspaces = [np.linspace(self.x_range[0], self.x_range[1], self.grid_points_per_dim) for _ in range(self.action_dim)]
        grid_mesh = np.meshgrid(*linspaces)
        self.X_grid = np.vstack([g.ravel() for g in grid_mesh]).T # Shape: (num_grid_points, action_dim)

        # 2. Compute prior covariance matrix K on the grid
        K = self._matern52_kernel(self.X_grid)

        # 3. Sample from N(0, K)
        try:
            # Add small jitter just in case kernel is ill-conditioned despite nugget
            # K += np.eye(self.num_grid_points) * 1e-8
            self._K_chol = cholesky(K, lower=True) # K = L @ L.T
            # Sample standard normal variables
            z = np.random.randn(self.num_grid_points)
            # Transform to sample from N(0, K): y = L @ z
            self.y_grid_sample = self._K_chol @ z
        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix was not positive definite. Using diagonal sampling (fallback).")
            # Fallback: sample independent values (not ideal, loses correlation)
            self.y_grid_sample = np.random.randn(self.num_grid_points) * self.sigma_f
            # Cannot compute inverse easily for prediction in this case
            self._K_chol = None # Mark as invalid
            self._K_inv_y = None

        # Precompute K_inv @ y_sample for faster predictions, if Cholesky succeeded
        if self._K_chol is not None:
             # Solve L @ v = y_sample
             v = solve_triangular(self._K_chol, self.y_grid_sample, lower=True)
             # Solve L.T @ alpha = v => alpha = K_inv @ y_sample
             self._K_inv_y = solve_triangular(self._K_chol.T, v, lower=False)
        else:
             self._K_inv_y = None # Cannot compute

        # 4. Estimate min/max/optimum FROM THE GRID SAMPLE
        self.max_y = np.max(self.y_grid_sample)
        self.min_y = np.min(self.y_grid_sample)
        optimum_idx = np.argmax(self.y_grid_sample)
        self.optimum_point = self.X_grid[optimum_idx, :]

        # Ensure max_y is strictly greater than min_y for scaling
        if np.isclose(self.max_y, self.min_y):
             self.min_y -= 1e-6 # Add small epsilon

        # print(f"GP Sample Initialized. Est. Min: {self.min_y:.3f}, Est. Max: {self.max_y:.3f} at {self.optimum_point}") # Debug
        return self.optimum_point, self.min_y, self.max_y

    def compute_y(self, x):
        """ Computes the GP predictive mean for the specific function sample. """
        if self.X_grid is None or self.y_grid_sample is None or self._K_inv_y is None:
             if self._K_chol is None and self.y_grid_sample is not None:
                 # Fallback case from initialize
                 print("Warning: Cannot compute GP prediction accurately due to failed Cholesky. Returning mean (0).")
                 # Return 0 or interpolate based on y_grid_sample (less accurate)
                 # For now, return 0 as we can't use the kernel structure
                 return np.zeros(np.atleast_2d(x).shape[0]).squeeze()
             else:
                 raise ValueError("GPSampler must be initialized before compute_y is called.")

        x = np.atleast_2d(x) # Ensure x is 2D

        # Compute kernel vector between new point(s) x and grid points X_grid
        # Do not add nugget/noise here
        k_x_Xgrid = self._matern52_kernel(x, self.X_grid)
        np.fill_diagonal(k_x_Xgrid, self.sigma_f**2) # Ensure diagonal is pure signal variance

        # Predictive mean E[f(x)|y_sample] = k(x, X_grid) @ K_inv @ y_sample
        y_pred = k_x_Xgrid @ self._K_inv_y

        return y_pred.squeeze() # Return scalar if single input, array if multiple