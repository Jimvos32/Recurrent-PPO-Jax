import numpy as np
from src.tasks.envs.function_envs.sampling_functions.base_sampler import FunctionSampler
import itertools

class BraninSampler(FunctionSampler):
    """
    Sampler for the Branin function (flipped for maximization).
    This function is defined for 2 dimensions ONLY.
    Formula: y(x1,x2) = -[ a(x2 - b*x1^2 + c*x1 - r)^2 + s(1-t)cos(x1) + s ]
    Standard domain: x1 in [-5, 10], x2 in [0, 15].
    Standard parameters (a, b, c, r, s, t) can be overridden via config.

    Initialization estimates min/max y within the bounds by sampling on a grid.
    """
    # Standard parameters
    _A = 1.0
    _B = 5.1 / (4.0 * np.pi**2)
    _C = 5.0 / np.pi
    _R = 6.0
    _S = 10.0
    _T = 1.0 / (8.0 * np.pi)
    _scaling = 1.0 # Scaling factor for the function value
    # Standard domain (can be list of tuples for per-dimension bounds)
    _STANDARD_X_RANGE = [(-5.0, 10.0), (0.0, 15.0)]
    
    # # Scaling factor for the function value (default is 1.0)
    # _A = 1.0
    # _B = 5.1 / (4.0 * np.pi**2)
    # _C = 5.0 / np.pi
    # _R = 6.0
    # _S = 44.81
    # _T = 1(10.0 / (8.0 * np.pi))
    # _scaling = 1.0 / 51.95 # Scaling factor for the function value
    # # Standard domain (can be list of tuples for per-dimension bounds)
    # _STANDARD_X_RANGE = [(0, 1), (0,1)]
    # Known global minima locations (for original function)
    _OPTIMA_LOCATIONS = np.array([
        [-np.pi, 12.275],
        [np.pi, 2.275],
        [9.42478, 2.475] # Approx 3*pi
    ])
    _OPTIMUM_VALUE = 0.397887 # Approx value at minima

    def __init__(self, action_dim, x_range=None, config=None):
        # Enforce 2 dimensions
        if action_dim != 2:
            print(f"Warning: Branin function is 2-dimensional. Overriding action_dim from {action_dim} to 2.")
            action_dim = 2

        x_range = [(-5.0, 10.0), (0.0, 15.0)]

        default_num_samples = 1000000 
        dim_samples = int(round(default_num_samples**(1/action_dim)))
        default_config = {
            "num_samples_per_dim": dim_samples, # Branin is only 2D, can afford more samples
            "a": self._A,
            "b": self._B,
            "c": self._C,
            "r": self._R,
            "s": self._S,
            "t": self._T,
            "scaling": self._scaling
        }
        if config:
            default_config.update(config)

        super().__init__(action_dim, x_range, default_config)

        # Store parameters
        self.a = self.config.get("a")
        self.b = self.config.get("b")
        self.c = self.config.get("c")
        self.r = self.config.get("r")
        self.s = self.config.get("s")
        self.t = self.config.get("t")
        self._scaling = self.config.get("scaling")

    def _get_bounds(self):
        """Helper to get lower and upper bounds per dimension."""
        lower_bounds = np.array([self.x_range[0][0], self.x_range[1][0]])
        upper_bounds = np.array([self.x_range[0][1], self.x_range[1][1]])
        return lower_bounds, upper_bounds

    def initialize(self):
        """
        Initializes the sampler by finding the min and max y values within the
        specified x_range using grid sampling, plus corners and known optima.
        Sets the optimum_point to the location of the maximum y value found
        (corresponding to the minimum of the original Branin function).
        """
        lower_bounds, upper_bounds = self._get_bounds()
        num_samples = self.config.get("num_samples_per_dim")

        # 1. Generate grid points respecting per-dimension bounds
        linspaces = [np.linspace(lower_bounds[i], upper_bounds[i], num_samples) for i in range(self.action_dim)]
        grid_points = np.array(list(itertools.product(*linspaces)))

        # 2. Generate corner points
        corners = np.array(list(itertools.product(*zip(lower_bounds, upper_bounds))))

        # 3. Identify known global optima locations within bounds
        optima_in_bounds_mask = np.all(
            (self._OPTIMA_LOCATIONS >= lower_bounds) & (self._OPTIMA_LOCATIONS <= upper_bounds),
            axis=1
        )
        optima_in_bounds = self._OPTIMA_LOCATIONS[optima_in_bounds_mask]

        # 4. Combine points: grid, corners, and in-bounds optima
        points_to_evaluate = [grid_points, corners]
        if optima_in_bounds.shape[0] > 0:
            points_to_evaluate.append(optima_in_bounds)

        all_points = np.vstack(points_to_evaluate)

        # 5. Remove duplicates
        unique_points = np.unique(all_points, axis=0)

        # 6. Evaluate the function at these unique points
        y_values = self.compute_y(unique_points)
        y_values = np.atleast_1d(y_values)

        # 7. Find min, max y, and the point corresponding to max y (min of original func)
        self.min_y = np.min(y_values) # Most negative value
        self.max_y = np.max(y_values) # Least negative value (closest to -0.397...)
        max_idx = np.argmax(y_values)
        self.optimum_point = unique_points[max_idx]

        # 8. Ensure max_y is strictly greater than min_y for scaling
        if np.isclose(self.max_y, self.min_y):
             self.min_y -= 1e-6 # Add small epsilon
             if np.isclose(self.max_y, self.min_y + 1e-6):
                 self.max_y += 1e-6

        return self.optimum_point, self.min_y, self.max_y

    def compute_y(self, x):
        """
        Compute the flipped Branin function value for input x.
        y(x1,x2) = -[ a(x2 - b*x1^2 + c*x1 - r)^2 + s(1-t)cos(x1) + s ]
        Input x should be shape (M, 2). Output shape is (M,).
        """
        x = np.atleast_2d(x)
        if x.shape[1] != self.action_dim: # Should always be 2 here
            raise ValueError(f"Input x must have {self.action_dim} columns (dimensions), but got shape {x.shape}")

        x1 = x[:, 0]
        x2 = x[:, 1]

        # Calculate term1: a(x2 - b*x1^2 + c*x1 - r)^2
        term_in_paren = x2 - self.b * x1**2 + self.c * x1 - self.r
        term1 = self.a * term_in_paren**2

        # Calculate term2: s(1-t)cos(x1)
        term2 = self.s * (1.0 - self.t) * np.cos(x1)

        # Calculate term3: s
        term3 = self.s

        # Original Branin function value
        f_val = term1 + term2 + term3
        
        f_val = self._scaling * f_val

        # Flip for maximization
        y = -f_val

        return y.squeeze()