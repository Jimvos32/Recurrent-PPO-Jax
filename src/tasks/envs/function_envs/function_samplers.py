# function_samplers.py
import numpy as np
import abc
# from src.tasks.envs.function_envs.sampling_functions.branin_sampler import BraninSampler
# from src.tasks.envs.function_envs.sampling_functions.eggholder_sampler import EggholderSamplerND
# # from src.tasks.envs.function_envs.sampling_functions.ackley_sampler import AckleySampler
# # from src.tasks.envs.function_envs.sampling_functions.cosine_sampler import CosineSampler
# # from src.tasks.envs.function_envs.sampling_functions.poly_sampler import PolySampler
# from src.tasks.envs.function_envs.sampling_functions.rosenbrock_sampler import RosenbrockSampler
# from src.tasks.envs.function_envs.sampling_functions.michalewicz_sampler import MichalewiczSampler
# from src.tasks.envs.function_envs.sampling_functions.hartmann_sampler import Hartmann6Sampler

class FunctionSampler(abc.ABC):
    """Abstract base class for function samplers."""
    def __init__(self, action_dim, x_range, config=None):
        self.action_dim = action_dim
        self.x_range = x_range
        self.config = config if config else {}
        self.optimum_point = None
        self.min_y = None
        self.max_y = None

    @abc.abstractmethod
    def initialize(self):
        """
        Initialize function parameters (e.g., coefficients, shifts)
        and determine optimum_point, min_y, max_y.
        Returns:
            tuple: (optimum_point, min_y, max_y)
        """
        pass

    @abc.abstractmethod
    def compute_y(self, x):
        """Compute the function value for input x."""
        pass

# --- Ackley Function Sampler ---
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

# --- Cosine Function Sampler ---
class CosineSampler(FunctionSampler):
    def __init__(self, action_dim, x_range, config=None):
        super().__init__(action_dim, x_range, config)
        # Default config values
        self.num_oscillations = self.config.get("num_oscillations", 3) # Example default
        self.c_bounds = self.config.get("c_bounds", (5.0, 20.0))
        self.A_bounds = self.config.get("A_bounds", (10.0, 20.0))
        self.B_bounds = self.config.get("B_bounds", (0.5, 1.5))
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

# --- Polynomial Function Sampler ---
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

        # Max value occurs at x_max
        self.max_y = self.c

        # Find minimum by checking the corners of the domain furthest from x_max
        lower, upper = self.x_range
        # Choose the bound (lower or upper) for each dimension that is furthest from x_max in that dimension
        x_min_coords = np.where(np.abs(self.x_max - lower) > np.abs(upper - self.x_max), lower, upper)
        self.min_y = self.compute_y(x_min_coords) # Use compute_y with initialized params

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
    


# --- Helper to get sampler class from name ---
def get_sampler_class(name):
    if name == 'ackley':
        return AckleySampler
    elif name == 'cosine':
        return CosineSampler
    elif name == 'poly':
        return PolySampler
    # elif name == 'eggholder':
    #     return EggholderSamplerND
    # elif name == 'rosenbrock':
    #     return RosenbrockSampler
    # elif name == 'michalewicz':
    #     return MichalewiczSampler
    # elif name == 'hartmann6':
    #     return Hartmann6Sampler
    # elif name == 'branin':
    #     return BraninSampler
        
   
    else:
        raise ValueError(f"Unknown function sampler name: {name}")