import sys
import jax
import jax.extend
import jaxlib
from jax.lib import xla_bridge

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("JAX version:", jax.__version__)
print("JAXLIB version:", jaxlib.__version__)
print("XLA_BACKEND:", jax.extend.backend.get_backend().platform)

