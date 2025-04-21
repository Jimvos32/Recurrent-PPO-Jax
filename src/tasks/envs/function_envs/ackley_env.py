from src.tasks.envs.base_fun_env import BaseOptimizationEnv
import numpy as np

class AckleyEnv(BaseOptimizationEnv):
    """
    Environment based on a flipped Ackley function adapted for maximization.
    
    The (flipped) Ackley function is defined as:
    
        f(x) = - [ -a * exp(-b * sqrt((1/n)*∑((x - s0)²))
                             - exp((1/n)*∑cos(c*(x - s0)))
                             + a + exp(1) ]
    
    which simplifies to:
    
        f(x) = a * exp(-b * sqrt((1/n)*∑((x - s0)²))
              + exp((1/n)*∑cos(c*(x - s0))) - a - exp(1)
    
    where:
      - n is the number of dimensions (action_dim)
      - s0 is the shift chosen to be the optimum point.
      
    At x = s0, we have f(x) = 0 (global maximum). Other values are negative.
    
    This environment also computes a tight approximation for the function bounds.
      - For 1D and 2D, the built-in find_empirical_bounds() function (with a dense grid) is used.
      - For dimensions >2, the function is evaluated at all corners of the input domain.
    """
    
    def __init__(self, env_config=None):
        
        self.a_bounds = env_config.get("a_bounds", (15.0, 20.0))
        self.b_bounds = env_config.get("b_bounds", (0.1, 0.2))
        self.c_bounds = env_config.get("c_bounds", (2 * np.pi, 2 * np.pi))
        super().__init__(env_config)
    
    def initialize_function(self):
        # Set Ackley parameters (use the values provided in env_config or defaults)
        self.a = np.random.uniform(self.a_bounds[0], self.a_bounds[1])  # Amplitude
        self.b =np.random.uniform(self.b_bounds[0], self.b_bounds[1])  # Frequency
        self.c = np.random.uniform(self.c_bounds[0], self.c_bounds[1])
        
        # Determine the optimum (shift) for the function.
        # Here we choose the optimum to lie within the central 60% of the domain:
        lower, upper = self.x_range
        self.optimum_point = np.random.uniform(lower + (upper - lower) * 0.2,
                                               upper - (upper - lower) * 0.2,
                                               size=(self.action_dim,))
        
        # The function value at the optimum should be the maximum.
        # With the flipped Ackley the optimum is at 0.
        self.max_y = self.compute_y(self.optimum_point)
        
        
        # For higher dimensions, evaluate the function at the corners of the hypercube.
        corners = np.array(np.meshgrid(*[[lower, upper] for _ in range(self.action_dim)]))
        # Reshape so that each row is a corner point.
        corners = corners.T.reshape(-1, self.action_dim)
        y_corners = self.compute_y(corners)
        self.min_y = np.min(y_corners)
    
    def compute_y(self, x):
        """
        Compute the flipped Ackley function value for input x.
        x can be a single point (shape: (action_dim,)) or a batch (shape: (n_points, action_dim)).
        
        The function is computed as:
            f(x) = a * exp(-b * sqrt((1/n)*∑((x - s0)²)))
                   + exp((1/n)*∑cos(c*(x - s0)))
                   - a - exp(1)
        and then flipped (multiplied by -1) so that its maximum (0) is attained at x = s0.
        """
        x = np.atleast_2d(x)
        n = self.action_dim
        
        # Shift the input so that the optimum (maximum) is at the origin.
        z = x - self.optimum_point
        
        # Compute squared-sum per sample.
        sum_sq = np.sum(z ** 2, axis=1)
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / n))
        
        # Compute the cosine sum per sample.
        term2 = -np.exp(np.sum(np.cos(self.c * z), axis=1) / n)
        
        # Compute the original Ackley function.
        f_val = term1 + term2 + self.a + np.exp(1)
        
        # Flip the function for maximization.
        y = -f_val
        return y.squeeze()
