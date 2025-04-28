from src.tasks.envs.function_envs.sampling_functions.base_sampler import FunctionSampler
import numpy as np
import jax

class PolySampler(FunctionSampler):
    def __init__(self, action_dim, x_range, config=None):
        super().__init__(action_dim, x_range, config)
        # Default config values
        self.degree = self.config.get("degree", 2)
        self.c_bounds = self.config.get("c_bounds", (5.0, 20.0))
        self.weight_bounds = self.config.get("weight_bounds", (0.5, 2.0)) # Renamed from x_bounds for clarity
        # Parameters to be set during initialize
        self.c = None
        self.weights = None
        self.x_max = None # This is the optimum point for this function

    def initialize(self):
        self.c = np.random.uniform(self.c_bounds[0], self.c_bounds[1])
        self.weights = np.random.uniform(self.weight_bounds[0], self.weight_bounds[1], size=(self.action_dim,))
        # Place optimum slightly inwards from boundaries
        self.x_max = np.random.uniform(self.x_range[0] * 0.9, self.x_range[1] * 0.9, size=(1, self.action_dim))
        self.optimum_point = self.x_max.squeeze() # Squeeze for consistency
        
        # jax.debug.print("Optimum point: {} {} {}", self.c, self.x_max, self.weights)

        # Max value occurs at x_max
        self.max_y = self.c

        # Find minimum by checking the corners of the domain furthest from x_max
        lower, upper = self.x_range
        # Choose the bound (lower or upper) for each dimension that is furthest from x_max in that dimension
        x_min_coords = np.where(np.abs(self.x_max - lower) > np.abs(upper - self.x_max), lower, upper)
        self.min_y = self.compute_y(x_min_coords) # Use compute_y with initialized params
        
        # values = np.linspace(self.min_y, self.max_y, num=1000) # Generate 100 values between min and max for scaling
        # c_max = -np.inf
        # for i in values:
        #     vv = self.compute_y(i)
        #     if vv > c_max:
        #         c_max = vv
        
        # jax.debug.print("Max value in range: {} min {} max {}", c_max, self.min_y, self.max_y)

        # Ensure max_y is strictly greater than min_y for scaling
        if np.isclose(self.max_y, self.min_y):
             self.min_y -= 1e-6 # Add small epsilon if min and max are too close

        return self.optimum_point, self.min_y, self.max_y

    def compute_y(self, x):
         # Ensure parameters are initialized
        if self.c is None or self.weights is None or self.x_max is None:
             raise ValueError("PolySampler must be initialized before compute_y is called.")

        x = np.atleast_2d(x)
        diff = x - self.x_max # x_max is shape (1, action_dim), broadcasts correctly
        # Ensure degree is even for a maximum at x_max, or adjust logic if odd degrees are needed
        if self.degree % 2 != 0:
            print(f"Warning: Polynomial degree {self.degree} is odd. The function might not have a maximum at x_max.")
            # Adjust calculation if needed, e.g., use absolute difference
            # weighted_term = self.weights * np.abs(diff ** self.degree) # Example for odd degrees

        # Assuming even degree for maximization form
        weighted_term = self.weights * (diff ** self.degree)
        result = self.c - np.sum(weighted_term, axis=1) # Max value is c, decreases away from x_max

        return result.squeeze()