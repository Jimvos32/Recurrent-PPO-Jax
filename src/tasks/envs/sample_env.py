import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax 
import jax.numpy as jnp

class SampleEnv(gym.Env):
    """
    A simple environment where the agent interacts with a second-degree polynomial.
    The agent's action corresponds to choosing an 'x' value, and the observation is the 
    corresponding 'y' value of the polynomial at that x.
    """
    def __init__(self, env_config=None, a=-1, b=10, x_range=(-10, 10)):
        super(SampleEnv, self).__init__()
        
        # Polynomial parameters: f(x) = -a*x^2 + b
        self.a = a
        self.b = b
        self.x_range = x_range  # Define the range of x values the agent can choose
        
        # Action space: Continuous, representing the x value that the agent chooses
        self.action_space = spaces.Box(low=np.array([self.x_range[0]]), high=np.array([self.x_range[1]]), shape=(1,), dtype=np.float32)
        
        # Observation space: Continuous, representing the y value (f(x)) at the chosen x
        self.observation_space = spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(1,), dtype=jnp.float32)
        
        # Initialize state
        self.state = None  # We don't need a specific state, just the observation of y
        self.x = None  # Agent's current x value

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start with a random x value within the specified range
        self.x = np.random.uniform(self.x_range[0], self.x_range[1])
        
        # Compute the corresponding y value using the polynomial function
        y = self.compute_y(self.x)
        
        # Set the initial state to the y value
        self.state = jnp.array([y], dtype=jnp.float32)
        
        return self.state, {}

    def step(self, action):
        """
        Step the environment by taking an action (selecting an x value).
        """
        
        # Ensure the action is within the valid range for x
        self.x = jnp.clip(action, self.x_range[0], self.x_range[1])
        
        # Update the x value with the agent's action
        # self.x = action[0]
        # print("x", self.x.shape)
        
        # Compute the corresponding y value using the polynomial function
        y = self.compute_y(self.x)
        
        # The reward is based on how close the agent's y is to the maximum y (at x=0)
        reward = -abs(self.compute_y(0) - y)  # Max y occurs at x = 0
        
        # The task is simple: the agent wants to maximize the y value, so it doesn't terminate
        done = False
        
        # Return the new observation (y value), reward, done flag, and additional info
        self.state = jnp.array([y], dtype=jnp.float32)
        # print("shaping the future", self.state.shape)
        
        
        metrics = {'final_info': {
                        'episode_length': 100,    # Numerical statistics
                        'total_reward': 150.5,
                        'success_rate': 0.85
                    },
                    'rewards': [1.0, 0.5, 2.0, -1.0],  # Array of rewards for the episode
                    'steps': 100,                # Additional numerical metrics
                    'progress': 0.75
                }
        
        return self.state, reward, done, False, metrics

    def compute_y(self, x):
        """
        Compute the y value of the polynomial at a given x.
        f(x) = -a*x^2 + b
        """
        return self.a * x**2 + self.b

    def render(self):
        """
        Render the environment for debugging purposes.
        """
        print(f"Current x: {self.x}, f(x): {self.state[0]}")