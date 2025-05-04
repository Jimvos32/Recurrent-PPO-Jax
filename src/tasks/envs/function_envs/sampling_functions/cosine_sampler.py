from src.tasks.envs.function_envs.sampling_functions.base_sampler import FunctionSampler
import numpy as np

class CosineSampler(FunctionSampler):
    def __init__(self, action_dim, x_range, config=None):
        super().__init__(action_dim, x_range, config)
        # Default config values
        self.num_oscillations = self.config.get("num_oscillations", 3) # Example default
        self.c_bounds = self.config.get("c_bounds", (1.0, 5.0))
        self.A_bounds = self.config.get("A_bounds", (5.0, 10.0))
        self.B_bounds = self.config.get("B_bounds", (1.0, 2.0))
        self.small_A_bounds = self.config.get("small_A_bounds", (0.2, 3.0))
        self.small_B_bounds = self.config.get("small_B_bounds", (0.5, 4.0))
        # Parameters to be set during initialize
        self.c = None
        self.A0 = None
        self.B0 = None
        self.s0 = None
        self.small_A = None
        self.small_B = None
        self.small_shift = None
        self.small_phase = None
        
        


    def initialize(self):
        self.c = np.random.uniform(self.c_bounds[0], self.c_bounds[1])
        self.A0 = np.random.uniform(self.A_bounds[0], self.A_bounds[1])
        self.B0 = np.random.uniform(self.B_bounds[0], self.B_bounds[1], size=(self.action_dim,))
        # Place shift (optimum) slightly inwards from boundaries
        self.s0 = np.random.uniform(self.x_range[0] * 0.9, self.x_range[1] * 0.9, size=(self.action_dim,))
        self.optimum_point = self.s0

        self.small_A = np.random.uniform(self.small_A_bounds[0], self.small_A_bounds[1], size=(self.action_dim, self.num_oscillations))
        self.small_B = np.random.uniform(self.small_B_bounds[0], self.small_B_bounds[1], size=(self.action_dim, self.num_oscillations))
        self.small_shift = np.tile(self.s0.reshape(-1, 1), (1, self.num_oscillations)) # Small oscillations centered around s0
        self.small_phase = np.zeros((self.action_dim, self.num_oscillations)) # Or random phase if needed

        # Theoretical bounds (can sometimes be loose, but avoids expensive search)
        self.max_y = self.c + self.A0 * self.action_dim + np.sum(self.small_A)
        self.min_y = self.c - self.A0 * self.action_dim - np.sum(self.small_A)
        
        # print("sdfh", self.num_oscillations)

        # Ensure max_y is strictly greater than min_y for scaling
        if np.isclose(self.max_y, self.min_y):
             self.min_y -= 1e-6 # Add small epsilon if min and max are too close


        return self.optimum_point, self.min_y, self.max_y

    def compute_y(self, x):
        # Ensure parameters are initialized
        if self.c is None or self.A0 is None or self.B0 is None or self.s0 is None or \
           self.small_A is None or self.small_B is None or self.small_shift is None or self.small_phase is None:
             raise ValueError("CosineSampler must be initialized before compute_y is called.")

        x = np.atleast_2d(x)
        result = np.full(x.shape[0], self.c)

        # Main cosine term
        for i in range(self.action_dim):
            result += self.A0 * np.cos(self.B0[i] * (x[:, i] - self.s0[i]))

        # Small oscillations
        for i in range(self.action_dim):
            for k in range(self.num_oscillations):
                diff = x[:, i] - self.small_shift[i, k]
                result += self.small_A[i, k] * np.cos(self.small_B[i, k] * diff + self.small_phase[i, k])

        return result.squeeze()
