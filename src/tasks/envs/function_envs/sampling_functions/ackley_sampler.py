from src.tasks.envs.function_envs.sampling_functions.base_sampler import FunctionSampler
import numpy as np
    

class AckleySampler(FunctionSampler):
    def __init__(self, action_dim, x_range, config=None):
        super().__init__(action_dim, x_range, config)
        # Default bounds, can be overridden by config
        self.a_bounds = self.config.get("a_bounds", (15.0, 20.0))
        self.b_bounds = self.config.get("b_bounds", (0.1, 0.2))
        self.c_bounds = self.config.get("c_bounds", (2 * np.pi, 2 * np.pi))
        # Parameters to be set during initialize
        self.a = None
        self.b = None
        self.c = None

    def initialize(self):
        self.a = np.random.uniform(self.a_bounds[0], self.a_bounds[1])
        self.b = np.random.uniform(self.b_bounds[0], self.b_bounds[1])
        self.c = np.random.uniform(self.c_bounds[0], self.c_bounds[1])

        lower, upper = self.x_range
        # Place optimum within the central 60% of the domain
        self.optimum_point = np.random.uniform(
            lower + (upper - lower) * 0.2,
            upper - (upper - lower) * 0.2,
            size=(self.action_dim,)
        )

        # Flipped Ackley's max is 0 at the optimum
        self.max_y = 0.0 # Theoretical max for flipped Ackley

        # Estimate min_y by checking corners
        corners = np.array(np.meshgrid(*[[lower, upper] for _ in range(self.action_dim)]))
        corners = corners.T.reshape(-1, self.action_dim)
        y_corners = self.compute_y(corners) # Use compute_y with initialized params
        self.min_y = np.min(y_corners)

        # Ensure max_y is strictly greater than min_y for scaling
        if np.isclose(self.max_y, self.min_y):
             self.min_y -= 1e-6 # Add small epsilon if min and max are too close


        return self.optimum_point, self.min_y, self.max_y

    def compute_y(self, x):
        # Ensure parameters are initialized before computing
        if self.a is None or self.b is None or self.c is None or self.optimum_point is None:
             raise ValueError("AckleySampler must be initialized before compute_y is called.")

        x = np.atleast_2d(x)
        n = self.action_dim
        z = x - self.optimum_point # Shift optimum to origin for calculation

        sum_sq = np.sum(z**2, axis=1)
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / n))
        term2 = -np.exp(np.sum(np.cos(self.c * z), axis=1) / n)

        f_val = term1 + term2 + self.a + np.exp(1)
        y = -f_val # Flip for maximization
        return y.squeeze()