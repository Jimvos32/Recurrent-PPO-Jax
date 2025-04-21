from src.tasks.envs.function_envs.function_samplers import FunctionSampler
import numpy as np
import itertools 

class Hartmann6Sampler(FunctionSampler):
    """
    Sampler for the Hartmann 6-dimensional function.
    Formula: y(x) = - sum_{i=1}^{4} [ alpha_i * exp( - sum_{j=1}^{6} A_ij * (x_j - P_ij)^2 ) ]
    This function is defined for 6 dimensions ONLY.
    Standard domain: [0, 1] for all dimensions.
    Parameters alpha, A, P are fixed constants, standard values are included
    but can be overridden via config. Global minimum (max y) is approx -3.32237.

    Initialization estimates min/max y within the bounds by sampling on a grid.
    Warning: Initialization cost is very high due to 6 dimensions.
    """
    # Standard parameters
    _STANDARD_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
    _STANDARD_A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    _STANDARD_P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    # Known global minimum location and value
    _OPTIMUM_LOC = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    _OPTIMUM_VALUE = -3.322368 # Max y-value (min of original function)

    def __init__(self, action_dim, x_range=(0.0, 1.0), config=None):
        # Enforce 6 dimensions
        if action_dim != 6:
            print(f"Warning: Hartmann6 function is 6-dimensional. Overriding action_dim from {action_dim} to 6.")
            action_dim = 6

        # Ensure x_range is a single tuple/list of length 2 for Hartmann6
        if not (isinstance(x_range, (tuple, list)) and len(x_range) == 2 and not isinstance(x_range[0], (tuple, list))):
             raise ValueError("x_range for Hartmann6Sampler must be a tuple or list of two elements (min, max).")

        # Default config - VERY LOW sample density due to 6D
        default_config = {
            "alpha": None, # Use standard if None
            "A": None,
            "P": None,
            "num_samples_per_dim": 5 # Results in 5^6 = 15625 samples
        }
        if config:
            default_config.update(config)

        super().__init__(action_dim, x_range, default_config)

        # Determine parameters alpha, A, P
        self.alpha = self.config.get("alpha")
        self.A = self.config.get("A")
        self.P = self.config.get("P")

        using_defaults = False
        if self.alpha is None and self.A is None and self.P is None:
             self.alpha = self._STANDARD_ALPHA
             self.A = self._STANDARD_A
             self.P = self._STANDARD_P
             using_defaults = True
        elif not (self.alpha is not None and self.A is not None and self.P is not None):
            raise ValueError("Config must provide all of 'alpha', 'A', 'P' or none (to use defaults).")

        # Validate provided parameters if not using defaults
        if not using_defaults:
             if not isinstance(self.alpha, np.ndarray): self.alpha = np.array(self.alpha)
             if not isinstance(self.A, np.ndarray): self.A = np.array(self.A)
             if not isinstance(self.P, np.ndarray): self.P = np.array(self.P)

             if self.alpha.shape != (4,): raise ValueError("Provided 'alpha' must have shape (4,).")
             if self.A.shape != (4, 6): raise ValueError("Provided 'A' must have shape (4, 6).")
             if self.P.shape != (4, 6): raise ValueError("Provided 'P' must have shape (4, 6).")

        # Print warning about sampling cost
        num_samples_per_dim = self.config.get("num_samples_per_dim")
        total_samples = num_samples_per_dim ** self.action_dim
        print(f"Warning: Hartmann6Sampler (N=6) initialization uses grid sampling "
              f"({num_samples_per_dim}^{self.action_dim} = {total_samples} points). "
              f"This can be very computationally expensive and memory intensive.")

    def initialize(self):
        """
        Initializes the sampler by finding the min and max y values within the
        specified x_range using grid sampling, plus corners and the known optimum.
        Sets the optimum_point to the location of the maximum y value found.
        """
        lower, upper = self.x_range # Single range [0, 1] for all dims
        num_samples = self.config.get("num_samples_per_dim")

        # 1. Generate grid points
        try:
            linspaces = [np.linspace(lower, upper, num_samples) for _ in range(self.action_dim)]
            grid_points = np.array(list(itertools.product(*linspaces)))
        except MemoryError:
             raise MemoryError(f"Failed to create grid ({num_samples}^{self.action_dim} points). "
                               f"Reduce 'num_samples_per_dim' for Hartmann6Sampler.")

        # 2. Generate corner points (2^6 = 64 corners)
        corners = np.array(list(itertools.product(*[[lower, upper] for _ in range(self.action_dim)])))

        # 3. Identify theoretical optimum location if within bounds
        optimum_loc = self._OPTIMUM_LOC
        optimum_in_bounds = np.all((optimum_loc >= lower) & (optimum_loc <= upper))

        # 4. Combine points: grid, corners, and optimum (if in bounds)
        points_to_evaluate = [grid_points, corners]
        if optimum_in_bounds:
            points_to_evaluate.append(optimum_loc.reshape(1, -1))

        all_points = np.vstack(points_to_evaluate)

        # 5. Remove duplicates
        unique_points = np.unique(all_points, axis=0)

        # 6. Evaluate the function at these unique points
        y_values = self.compute_y(unique_points)
        y_values = np.atleast_1d(y_values)

        # 7. Find min, max y, and the point corresponding to max y
        self.min_y = np.min(y_values) # Most negative value
        self.max_y = np.max(y_values) # Least negative value (closest to -3.32...)
        max_idx = np.argmax(y_values)
        self.optimum_point = unique_points[max_idx]

        # 8. Ensure max_y is strictly greater than min_y for scaling
        if np.isclose(self.max_y, self.min_y):
             self.min_y -= 1e-6
             if np.isclose(self.max_y, self.min_y + 1e-6):
                 self.max_y += 1e-6

        # Ensure max_y does not exceed theoretical maximum (min of original func)
        self.max_y = min(self.max_y, self._OPTIMUM_VALUE)
        # Ensure min_y <= max_y after capping
        self.min_y = min(self.min_y, self.max_y - 1e-6 if np.isclose(self.min_y, self.max_y) else self.min_y)


        return self.optimum_point, self.min_y, self.max_y

    def compute_y(self, x):
        """
        Compute the Hartmann 6-dimensional function value for input x.
        y(x) = - sum_{i=0}^{3} [ alpha_i * exp( - sum_{j=0}^{5} A_ij * (x_j - P_ij)^2 ) ]
        Input x should be shape (M, 6). Output shape is (M,).
        """
        x = np.atleast_2d(x)
        if x.shape[1] != self.action_dim: # Should always be 6
            raise ValueError(f"Input x must have {self.action_dim} columns (dimensions), but got shape {x.shape}")

        num_points = x.shape[0]
        total_y = np.zeros(num_points)

        # Outer sum over i = 0 to 3
        for i in range(4):
            alpha_i = self.alpha[i]
            A_i = self.A[i, :] # Shape (6,)
            P_i = self.P[i, :] # Shape (6,)

            # Calculate inner sum: sum_{j=0}^{5} A_ij * (x_j - P_ij)^2
            # x is (M, 6), P_i is (6,). Reshape P_i to (1, 6) for broadcasting.
            # A_i is (6,). Reshape A_i to (1, 6) for broadcasting element-wise product.
            diff_sq = (x - P_i.reshape(1, -1))**2 # Shape (M, 6)
            term_in_sum = A_i.reshape(1, -1) * diff_sq # Shape (M, 6)
            inner_sum = np.sum(term_in_sum, axis=1) # Shape (M,)

            # Calculate exponential term for outer sum
            exp_term = alpha_i * np.exp(-inner_sum)

            # Accumulate the sum (note the negative sign in the overall formula)
            total_y -= exp_term

        return total_y.squeeze()