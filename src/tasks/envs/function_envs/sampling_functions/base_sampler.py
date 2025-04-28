import abc

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