import numpy as np
from src.tasks.envs.function_envs.sampling_functions.base_sampler import FunctionSampler
import itertools

class RosenbrockSampler(FunctionSampler):
    """
    Sampler for the Rosenbrock function (flipped for maximization).
    Formula: y(x) = - sum_{i=0}^{N-2} [ 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
    For N = 1: Returns 0.
    Standard domain: [-5, 10] for all dimensions.
    Global maximum (of flipped func) is 0 at (1, 1, ..., 1).

    Initialization estimates min/max y within the bounds by sampling on a grid.
    Warning: Initialization cost scales exponentially with action_dim for N >= 2.
    """
    _THEORETICAL_OPTIMUM_VALUE = 0.0

    def __init__(self, action_dim, x_range=(-5.0, 10.0), config=None):
        if action_dim < 1:
            raise ValueError("RosenbrockSampler requires action_dim >= 1.")
        # Ensure x_range is a single tuple/list of length 2 for Rosenbrock
        if not (isinstance(x_range, (tuple, list)) and len(x_range) == 2 and not isinstance(x_range[0], (tuple, list))):
             raise ValueError("x_range for RosenbrockSampler must be a tuple or list of two elements (min, max).")
         
        self.x_range = [-5,10]
        x_range = self.x_range

        # Default config for grid sampling density
        default_num_samples = 1000000 
        dim_samples = int(round(default_num_samples**(1/action_dim)))
        default_config = {"num_samples_per_dim": dim_samples}
        if config:
            default_config.update(config)

        super().__init__(action_dim, x_range, default_config)
        
        

        if self.action_dim >= 2:
             num_samples_per_dim = self.config.get("num_samples_per_dim")
             total_samples = num_samples_per_dim ** self.action_dim
             print(f"Warning: RosenbrockSampler (N={self.action_dim}) initialization uses grid sampling "
                   f"({num_samples_per_dim}^{self.action_dim} = {total_samples} points). "
                   f"This can be computationally expensive.")

        # Location of the optimum for the flipped function
        self._theoretical_optimum_point = np.ones(action_dim)

    def initialize(self):
        """
        Initializes the sampler.
        For N = 1: Sets min/max to 0 and optimum to 1 (if in range) or midpoint.
        For N >= 2: Finds min/max y within the specified x_range using grid
                    sampling, plus corners and the theoretical optimum (1, ..., 1).
                    Sets optimum_point to the location of the maximum y value found.
        """
        lower, upper = self.x_range # Single range for all dims

        if self.action_dim == 1:
            self.min_y = 0.0
            self.max_y = 0.0
            # Optimum is 1.0 for N=1 (term (1-x0)^2, min at x0=1) -> max 0 for flipped
            if lower <= 1.0 <= upper:
                self.optimum_point = np.array([1.0])
            else:
                # If 1 is out of bounds, the optimum within bounds depends on the range.
                # The function -(1-x)^2 decreases away from 1.
                # Max value will be at the bound closest to 1.
                if abs(lower - 1.0) < abs(upper - 1.0):
                    self.optimum_point = np.array([lower])
                else:
                    self.optimum_point = np.array([upper])
                # We still set max_y = 0, min_y = -epsilon, as the function is <= 0
                self.max_y = self.compute_y(self.optimum_point) # Should be <= 0

        else: # action_dim >= 2
            num_samples = self.config.get("num_samples_per_dim")
            # 1. Generate grid points
            try:
                linspaces = [np.linspace(lower, upper, num_samples) for _ in range(self.action_dim)]
                grid_points = np.array(list(itertools.product(*linspaces)))
            except MemoryError:
                 raise MemoryError(f"Failed to create grid ({num_samples}^{self.action_dim} points). "
                                   f"Reduce 'num_samples_per_dim' or action_dim for RosenbrockSampler.")

            # 2. Generate corner points
            corners = np.array(list(itertools.product(*[[lower, upper] for _ in range(self.action_dim)])))

            # 3. Identify theoretical optimum location (1, ..., 1) if within bounds
            optimum_loc = self._theoretical_optimum_point
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
            self.max_y = np.max(y_values) # Least negative value (closest to 0)
            max_idx = np.argmax(y_values)
            self.optimum_point = unique_points[max_idx]

        # 8. Ensure max_y is strictly greater than min_y for scaling
        if np.isclose(self.max_y, self.min_y):
             # Add epsilon to min_y first
             self.min_y -= 1e-6
             # If max_y was also the same, adjust it to be slightly positive relative to new min_y
             if np.isclose(self.max_y, self.min_y + 1e-6):
                  self.max_y += 1e-6 # Ensure max_y > min_y

        # Ensure max_y is not positive (it should be <= 0 for flipped Rosenbrock)
        self.max_y = min(self.max_y, 0.0)
        # Ensure min_y <= max_y after capping max_y
        self.min_y = min(self.min_y, self.max_y - 1e-6 if np.isclose(self.min_y, self.max_y) else self.min_y)


        return self.optimum_point, self.min_y, self.max_y

    def compute_y(self, x):
        """
        Compute the flipped Rosenbrock function value for input x.
        y(x) = - sum_{i=0}^{N-2} [ 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
        For N = 1, y(x) = -(1 - x_0)^2.
        Input x should be shape (M, N). Output shape is (M,).
        """
        x = np.atleast_2d(x)
        if x.shape[1] != self.action_dim:
            raise ValueError(f"Input x must have {self.action_dim} columns (dimensions), but got shape {x.shape}")

        if self.action_dim == 1:
            x0 = x[:, 0]
            f_val = (1.0 - x0)**2
        else: # N >= 2
            # Calculate sum using slicing
            xi = x[:, :-1] # Shape (M, N-1) - selects columns 0 to N-2
            xi_plus_1 = x[:, 1:] # Shape (M, N-1) - selects columns 1 to N-1

            term1 = 100.0 * (xi_plus_1 - xi**2)**2
            term2 = (1.0 - xi)**2
            # Sum over the N-1 terms for each point (sum across columns axis=1)
            f_val = np.sum(term1 + term2, axis=1) # Shape (M,)

        # Flip for maximization
        y = -f_val

        return y.squeeze()