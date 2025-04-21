# env_state.py
from flax import struct
import jax.numpy as jnp
import jax.random
from typing import Any, Tuple, Dict

@struct.dataclass
class EnvState:
    """Holds the state of the MultiFunctionEnv."""
    key: jax.random.PRNGKey
    tick: int
    # --- History / RL specific state ---
    last_scaled_avg_obs: jnp.ndarray
    best_scaled_obs_so_far: jnp.ndarray # Track best single point found

    # --- Function specific state ---
    function_type_index: int # 0: Ackley, 1: Cosine, 2: Poly
    function_params: Any # Holds AckleyParams, CosineParams, or PolyParams

    # --- Current Episode Config ---
    current_batch_size: int # Store the batch size selected for this episode << NEW

    # --- Static Env Config (copied for use in jitted functions) ---
    x_range: Tuple[float, float]
    action_dim: int
    max_batches: int
    reward_config: Dict[str, float] # Assuming float values in reward dict
    max_episode_steps: int