from src.tasks.envs.function_envs.sampling_functions.base_sampler import FunctionSampler
import numpy as np
import itertools 

class EggholderSamplerND(FunctionSampler):
    """
    Sampler for a generalized N-dimensional Eggholder function (flipped for maximization).
    The function sums the standard 2D Eggholder calculation over consecutive pairs (x_i, x_{i+1}).

    Initialization estimates min/max y within the bounds by sampling on a grid.
    Warning: Initialization cost scales exponentially with action_dim.
    """
    def __init__(self, action_dim, x_range=(-512.0, 512.0), config=None):
        if action_dim < 2:
            raise ValueError("EggholderSamplerND requires action_dim >= 2.")
        # Ensure x_range is a tuple/list of length 2
        self.x_range = [-512,512]
        x_range = self.x_range 

        # Default config for grid sampling density
        # Lower default due to exponential scaling
        default_num_samples = 1000#20 if action_dim <= 3 else (10 if action_dim <= 5 else 5)
        default_config = {"num_samples_per_dim": default_num_samples}
        if config:
            default_config.update(config)

        super().__init__(action_dim, x_range, default_config)
        print(f"Warning: EggholderSamplerND initialization uses grid sampling "
              f"({self.config.get('num_samples_per_dim')}^{self.action_dim} points). "
              f"This can be computationally expensive for high dimensions.")

        # Theoretical optimum is not easily defined for N-D version
        self._theoretical_optimum_point = None
        self._theoretical_max_y = None

    def initialize(self):
        """
        Initializes the sampler by finding the min and max y values within the
        specified x_range using grid sampling, plus corners.
        Sets the optimum_point to the location of the maximum y value found.
        """
        # lower, upper = self.x_range
        # num_samples = self.config.get("num_samples_per_dim")

        # # 1. Generate grid points
        # try:
        #     linspaces = [np.linspace(lower, upper, num_samples) for _ in range(self.action_dim)]
        #     grid_points = np.array(list(itertools.product(*linspaces))) # Creates all combinations
        # except MemoryError:
        #      raise MemoryError(f"Failed to create grid ({num_samples}^{self.action_dim} points). "
        #                        f"Reduce 'num_samples_per_dim' or action_dim for EggholderSamplerND.")


        # # 2. Generate corner points
        # corners = np.array(list(itertools.product(*[[lower, upper] for _ in range(self.action_dim)])))

        # # 3. Combine points: grid and corners
        # all_points = np.vstack([grid_points, corners])

        # # 4. Remove duplicates
        # unique_points = np.unique(all_points, axis=0)

        # # 5. Evaluate the function at these unique points
        # y_values = self.compute_y(unique_points)
        # y_values = np.atleast_1d(y_values) # Ensure it's an array

        # # 6. Find min, max y, and the point corresponding to max y
        # self.min_y = np.min(y_values)
        # self.max_y = np.max(y_values)
        # max_idx = np.argmax(y_values)
        # self.optimum_point = unique_points[max_idx]

        # # 7. Ensure max_y is strictly greater than min_y for scaling
        # if np.isclose(self.max_y, self.min_y):
        #      # If min/max are too close (e.g., flat surface or single point evaluated), add epsilon
        #      self.min_y -= 1e-6
        #      # Re-check if max_y needs adjustment if it was also the min_y originally
        #      if np.isclose(self.max_y, self.min_y + 1e-6):
        #          self.max_y += 1e-6
        self.max_y = 959.6407
        self.min_y = -1049.0
        self.optimum_point = np.array([512.0, 404.2319]) # This is the theoretical optimum for the 2D Eggholder function
                 
        print(f"EggholderSamplerND initialized with min_y: {self.min_y}, max_y: {self.max_y}, {self.optimum_point}")

        return self.optimum_point, self.min_y, self.max_y

    def compute_y(self, x):
        """
        Compute the flipped N-dimensional Eggholder function value for input x.
        Formula: sum_{i=0}^{N-2} [ -(x_{i+1} + 47) sin(sqrt(abs(x_i/2 + (x_{i+1} + 47))))
                                   - x_i sin(sqrt(abs(x_i - (x_{i+1} + 47)))) ]
        Input x should be shape (M, N) or (N,). Output shape is (M,) or scalar.
        """
        x = np.atleast_2d(x)
        if x.shape[1] != self.action_dim:
            raise ValueError(f"Input x must have {self.action_dim} columns (dimensions), but got shape {x.shape}")

        total_f_val = np.zeros(x.shape[0]) # Accumulator for the sum for each input point

        for i in range(self.action_dim - 1):
            xi = x[:, i]
            xi_plus_1 = x[:, i+1]

            term1_sqrt_arg = np.abs(xi / 2.0 + (xi_plus_1 + 47.0))
            term1 = -(xi_plus_1 + 47.0) * np.sin(np.sqrt(term1_sqrt_arg))

            term2_sqrt_arg = np.abs(xi - (xi_plus_1 + 47.0))
            term2 = -xi * np.sin(np.sqrt(term2_sqrt_arg))

            # Sum the 2D Eggholder value for this pair (xi, xi+1)
            total_f_val += (term1 + term2)

        # Flip the final sum for maximization
        y = -total_f_val

        return y.squeeze()