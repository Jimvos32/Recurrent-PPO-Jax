import jax, jax.numpy as jnp
from src.agents.ppo_dic_inherits.inh_agents.standard_sampling import SamplingImplBase
from src.agents.ppo_dic_inherits.inh_agents.standard_sampling_jax import SamplingImplBaseJax
from flowjax.distributions import MultivariateNormal, Transformed, Normal
from flowjax.bijections import Tanh, Chain, AbstractBijection


