import flax.linen as nn
import jax.numpy as jnp
import jax

from typing import Callable
from src.utils import tree_index
from flax.linen.initializers import constant, orthogonal
import numpy as np


class ActorCriticModel(nn.Module):
    repr_model_fn:Callable
    seq_model_fn:Callable
    actor_fn:Callable
    critic_fn:Callable


    def setup(self):
        self.repr_model=self.repr_model_fn()
        self.seq_model=self.seq_model_fn()
        self.actor=self.actor_fn()
        self.critic=self.critic_fn()
    
    @nn.compact
    def __call__(self,inputs,terminations,last_memory):  
        """_summary_

        Args:
            inputs (_type_): shape (TXrepr_dim)
            terminations: (T)
            last_memory (_type_): as required by seq_model

        Returns:
            _type_: _description_
        """
        
       
        rep = self.repr_model(inputs)
        # TXlatent_dim, image or otherwise, they are always flattened
        # print("rep", rep.shape)
        rep=rep.reshape(rep.shape[0],-1)
        rep = jnp.concatenate([rep, inputs["step"]], axis=1)
        # print("rep2", rep.shape, terminations.shape, last_memory[0][0].shape)
        seq_rep,memory=self.seq_model(rep,terminations,last_memory)
        # print("seq_rep", seq_rep.shape, memory[0][0].shape)
        
        seq_rep=jnp.concatenate([seq_rep, inputs["step"]], axis=1)
        # print("seq_rep2", seq_rep.shape)
        actor_out=self.actor(seq_rep)
        # print("totalinp", inputs.shape, "actor_in", seq_rep.shape, "actor_out", actor_out.shape)
        # print("actor_out", actor_out.shape, inputs["step"].shape, inputs["reward"].shape)
        # means = actor_out.shape[-1] // 2
        # actor_out = actor_out.at[:, :means].set(jnp.full_like(actor_out[:, :means], inputs["reward"]))
        critic_out=self.critic(seq_rep)
        # jax.debug.print("actor  out: \n{}\n", actor_out[0])
        # print(actor_out.shape, critic_out.shape)
        # jax.debug.print("we are called in seq {} {}", last_memory[0][0][0], inputs["step"])
        return actor_out,critic_out,memory
    
    
class ActorCriticVAEModel(nn.Module):
    repr_model_fn:Callable
    seq_model_fn:Callable
    actor_fn:Callable
    critic_fn:Callable


    def setup(self):
        self.repr_model=self.repr_model_fn()
        self.seq_model=self.seq_model_fn()
        self.actor=self.actor_fn()
        self.critic=self.critic_fn()
    
    @nn.compact
    def __call__(self,inputs,terminations,last_memory):  
        """_summary_

        Args:
            inputs (_type_): shape (TXrepr_dim)
            terminations: (T)
            last_memory (_type_): as required by seq_model

        Returns:
            _type_: _description_
        """
       
        rep = self.repr_model(inputs)
        # TXlatent_dim, image or otherwise, they are always flattened
        # print("rep", rep.shape)
        rep=rep.reshape(rep.shape[0],-1)
        rep = jnp.concatenate([rep, inputs["step"]], axis=1)
        # print("rep2", rep.shape, terminations.shape, last_memory[0][0].shape)
        seq_rep,memory=self.seq_model(rep,terminations,last_memory)
        # print("seq_rep", seq_rep.shape, memory[0][0].shape)
        seq_rep=jnp.concatenate([seq_rep, inputs["step"]], axis=1)
        # print("seq_rep2", seq_rep.shape)
        actor_out, latent_vars = self.actor(seq_rep)
        # print("totalinp", inputs.shape, "actor_in", seq_rep.shape, "actor_out", actor_out.shape)
        critic_out=self.critic(seq_rep)
        # print(actor_out.shape, critic_out.shape)
        return actor_out,critic_out,memory, latent_vars

