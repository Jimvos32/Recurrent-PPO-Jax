from src.tasks.envs.function_envs.function_samplers import FunctionSampler
import numpy as np
import itertools 


class MichalewiczSampler(FunctionSampler):
    """
    Sampler for the Michalewicz function (flipped for maximization).
    Formula: y(x) = sum_{i=1}^{N} [ sin(x_i) * (sin( (i+1) * x_i^2 / pi ))^(2m) ]
    Standard domain: [0, pi] for all dimensions.
    Parameter 'm' controls steepness (default m=10).

    Initialization estimates min/max y within the bounds by sampling on a grid.
    Warning: Initialization cost scales exponentially with action_dim.
    """
    def __init__(self, action_dim, x_range=(0.0, np.pi), config=None):
        if action_dim < 1:
            raise ValueError("MichalewiczSampler requires action_dim >= 1.")
        # Ensure x_range is a tuple/list of length 2
        if not (isinstance(x_range, (tuple, list)) and len(x_range) == 2):
             raise ValueError("x_range must be a tuple or list of two elements (min, max).")

        # Default config for m and grid sampling density
        default_m = 10.0
        # Adjust sampling based on dim, similar to EggholderND
        default_num_samples = 20 if action_dim <= 3 else (10 if action_dim <= 5 else 5)
        default_config = {
            "m": default_m,
            "num_samples_per_dim": default_num_samples
        }
        if config:
            default_config.update(config) # Overwrite defaults with user config

        super().__init__(action_dim, x_range, default_config)

        self.m = self.config.get("m")

        if self.action_dim >= 1: # Always print warning if sampling is used
             num_samples_per_dim = self.config.get("num_samples_per_dim")
             total_samples = num_samples_per_dim ** self.action_dim
             print(f"Warning: MichalewiczSampler (N={self.action_dim}) initialization uses grid sampling "
                   f"({num_samples_per_dim}^{self.action_dim} = {total_samples} points). "
                   f"This can be computationally expensive.")
        # Theoretical optimum is hard to find/not fixed
        self._theoretical_optimum_point = None
        self._theoretical_max_y = None # Max of flipped function (min of original)

    def initialize(self):
        """
        Initializes the sampler by finding the min and max y values within the
        specified x_range using grid sampling, plus corners.
        Sets the optimum_point to the location of the maximum y value found.
        """
        lower, upper = self.x_range
        num_samples = self.config.get("num_samples_per_dim")

        # 1. Generate grid points
        try:
            linspaces = [np.linspace(lower, upper, num_samples) for _ in range(self.action_dim)]
            grid_points = np.array(list(itertools.product(*linspaces))) # Creates all combinations
        except MemoryError:
             raise MemoryError(f"Failed to create grid ({num_samples}^{self.action_dim} points). "
                               f"Reduce 'num_samples_per_dim' or action_dim for MichalewiczSampler.")

        # 2. Generate corner points
        corners = np.array(list(itertools.product(*[[lower, upper] for _ in range(self.action_dim)])))

        # 3. Combine points: grid and corners
        all_points = np.vstack([grid_points, corners])

        # 4. Remove duplicates
        unique_points = np.unique(all_points, axis=0)

        # 5. Evaluate the function at these unique points
        y_values = self.compute_y(unique_points)
        y_values = np.atleast_1d(y_values) # Ensure it's an array

        # 6. Find min, max y, and the point corresponding to max y
        self.min_y = np.min(y_values)
        self.max_y = np.max(y_values)
        max_idx = np.argmax(y_values)
        self.optimum_point = unique_points[max_idx]

        # 7. Ensure max_y is strictly greater than min_y for scaling
        if np.isclose(self.max_y, self.min_y):
             self.min_y -= 1e-6 # Add small epsilon
             if np.isclose(self.max_y, self.min_y + 1e-6):
                 self.max_y += 1e-6

        return self.optimum_point, self.min_y, self.max_y

    def compute_y(self, x):
        """
        Compute the flipped Michalewicz function value for input x.
        y(x) = sum_{i=0}^{N-1} [ sin(x_i) * (sin( (i+1) * x_i^2 / pi ))^(2m) ]
        Input x should be shape (M, N). Output shape is (M,).
        """
        x = np.atleast_2d(x)
        if x.shape[1] != self.action_dim:
            raise ValueError(f"Input x must have {self.action_dim} columns (dimensions), but got shape {x.shape}")

        # Initialize sum for each input point
        total_y = np.zeros(x.shape[0])

        # Sum over dimensions i=0 to N-1 (corresponding to i=1 to N in formula)
        for i in range(self.action_dim):
            xi = x[:, i]
            # Calculate term inside sin^(2m): (i+1) * xi^2 / pi
            # Using i+1 because loop index i starts from 0
            term_in_sin = (i + 1.0) * (xi**2) / np.pi

            # Calculate sin(xi) * (sin(term_in_sin))^(2m)
            # Need to handle potential numerical issues if sin(term_in_sin) is negative,
            # although raising to an even power (2m) should make it positive.
            # Using np.power is generally safe.
            sin_term_pow_2m = np.power(np.sin(term_in_sin), 2 * self.m)
            term_i = np.sin(xi) * sin_term_pow_2m

            total_y += term_i

        return total_y.squeeze()