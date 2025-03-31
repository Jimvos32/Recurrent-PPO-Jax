# ============================
# File: sampling_parent.py
# ============================
import abc
import jax
import jax.numpy as jnp
import numpy as np
import logging
import tqdm
from argparse import Namespace
from typing import Callable
import rlax

# Assume your actor-critic model and helper (e.g. nn.vmap) are imported from your modules:
from src.models.actor_critic import ActorCriticModel, nn

logger = logging.getLogger(__name__)

# Helper functions (same as before)
def jax_to_numpy(*args):
    return jax.tree_map(lambda x: np.array(x), args)

def numpy_to_jax(*args, dtype=jnp.float32):
    return jax.tree_map(lambda x: jnp.array(x, dtype=dtype), args)

class RootAgent:
    """
    This class holds all the common agent logic (reset, unroll, evaluate, etc.)
    It does not itself implement the sampling functions; instead, it expects
    a sampling_impl dictionary to be passed in that defines:
       - sampling_differ(act_logits, random_key, masks, agent)
       - gaussian_log_prob(actions, act_logits, agent)
       - entropy(logits, agent)
    The extra parameter "agent" is provided so that the functions can access properties
    such as the environment.
    """
    def __init__(self,
                 train_envs,
                 eval_env,
                 rollout_len,
                 repr_model_fn: Callable,
                 seq_model_fn: Callable,
                 actor_fn: Callable,
                 critic_fn: Callable,
                 sampling_impl_class,
                 sequence_length=None,
                 single_dim=False,
                 task_name=None
            ):
        self.env = train_envs
        self.eval_env = eval_env
        self.rollout_len = rollout_len
        if sequence_length is None:
            self.sequence_length = rollout_len
        else:
            assert rollout_len % sequence_length == 0, "rollout_len must be divisible by sequence_length"
            self.sequence_length = sequence_length
        self.seq_fn, self.seq_init = seq_model_fn
        self.single_dim = single_dim
        self.task = task_name
        max_batch = self.eval_env.unwrapped.max_batches
        action_dim = self.eval_env.unwrapped.action_dim
        self.sampling_impl = sampling_impl_class(action_dim=action_dim, max_batch=max_batch)
        self.sampling_impl.agent = self  # give access to env

        # Build the actor-critic model
        self.ac_model = nn.vmap(ActorCriticModel,
                                variable_axes={'params': None},
                                split_rngs={'params': False, 'vae_sample': True})(
                                    repr_model_fn, self.seq_fn, actor_fn, critic_fn)

        @jax.jit
        def actor_critic_fn(random_key, params, inputs, terminations, last_memory):
            # jax.debug.print("is this thei first {} {} ", terminations.shape, inputs[inputs])
            random_key, vae_sample_key = jax.random.split(random_key)
            
        
            if terminations.shape == ():
                terminations = jnp.expand_dims(jnp.expand_dims(terminations, 0), 0)
            
            act_logits, values, memory = self.ac_model.apply(
                params, inputs, terminations, last_memory,
                rngs={'random': random_key, 'vae_sample': vae_sample_key})
            
           
            
            # jax.debug.print("act_logits {} values {} memory{}, inputs {} terminations {}", 
            #                 act_logits.shape, values.shape, memory[0][0].shape, inputs["step"].shape, terminations.shape)
            return act_logits, values, memory

        self.actor_critic_fn = actor_critic_fn

    def reset(self, params_key, random_key):
        self.tick = 0
        self.o_tick, _ = self.env.reset()
        self.r_tick = jnp.zeros(self.env.num_envs)
        self.term_tick = jnp.full((self.env.num_envs,), False)
        
        self.h_tickminus1 = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0),
                                                              self.env.num_envs, axis=0),
                                         self.seq_init())
        expanded_o = self.expand_o_tick(self.o_tick)
        
        # print("reset error ", self.r_tick.shape, self.term_tick.shape, jnp.expand_dims(self.term_tick, 1))
        
        self._params = self.ac_model.init({'params': params_key, 'random': random_key},
                                          expanded_o,
                                          jnp.expand_dims(self.term_tick, 1),
                                          self.h_tickminus1)
        def params_sum(params):
            return sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params)))
        logger.info("Total Number of params: %d" % params_sum(self.params))
        
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, value):
        self._params = value

    def expand_o_tick(self, o_tick, eval=False):
        if eval:
            return {k: jnp.expand_dims(v, axis=0) for k, v in o_tick.items()}
        return {k: jnp.expand_dims(v, axis=1) for k, v in o_tick.items()}
    
    def stack_dict_obs(self, obs, dictio):
        if len(dictio.keys()) == 0:
            return {k: np.expand_dims(np.array(v), axis=1) for k, v in obs.items()}
        else:
            for k, v in obs.items():
                if k in dictio:
                    dictio[k] = np.concatenate([dictio[k], np.expand_dims(np.array(v), axis=1)], axis=1)
                else:
                    dictio[k] = jnp.expand_dims(v, axis=1)
            return dictio

    def unroll_actors(self, random_key):
        num_seqs = self.rollout_len // self.sequence_length
        h_tickminus1 = self.h_tickminus1
        o_tick = self.o_tick
        r_tick = self.r_tick
        term_tick = self.term_tick
        actions = []
        observations = {}
        rewards = []
        critic_preds = []
        actor_preds = []
        terminations = []
        hiddens = []
        masked = []
        hidden_indices = []
        infos = []
        
        
        for t in range(self.rollout_len):
            observations = self.stack_dict_obs(o_tick, observations)
            rewards.append(r_tick.copy())
            terminations.append(term_tick.copy())
            random_key, model_key = jax.random.split(random_key)
            if t % self.sequence_length == 0:
                hiddens.append(jax.tree_map(lambda x: x, h_tickminus1))
                hidden_indices.append(jnp.repeat(jnp.arange(t, t+self.sequence_length).reshape(1, -1),
                                                 repeats=self.env.num_envs, axis=0))
            expanded_o = self.expand_o_tick(o_tick)
            act_logits, v_tick, htick = self.actor_critic_fn(model_key, self.params,
                                                               expanded_o,
                                                               jnp.expand_dims(term_tick, 1),
                                                               h_tickminus1)
            masks = expanded_o.get("mask", None)
           
            acts_tick = self.sampling_impl.sampling_differ(act_logits, random_key, masks)
            o_tickplus1, r_tickplus1, term_tickplus1, trunc_tickplus1, info = self.env.step(*jax_to_numpy(acts_tick))
            o_tickplus1, r_tickplus1 = numpy_to_jax(o_tickplus1, r_tickplus1)
            term_tickplus1, trunc_tickplus1 = numpy_to_jax(term_tickplus1, trunc_tickplus1, dtype=bool)
            term_tickplus1 = jnp.logical_or(term_tickplus1, trunc_tickplus1)
            critic_preds.append(v_tick.copy())
            actor_preds.append(act_logits.copy())
            actions.append(acts_tick.copy())
            infos.append(info)
            masked.append(masks)
            o_tick = o_tickplus1
            r_tick = r_tickplus1
            h_tickminus1 = htick
            term_tick = term_tickplus1
            self.tick += 1
        observations = self.stack_dict_obs(o_tick, observations)
        rewards.append(r_tick)
        terminations.append(term_tick)
        random_key, model_key = jax.random.split(random_key)
        expanded_o = self.expand_o_tick(o_tick)
        _, v_tick, _ = self.actor_critic_fn(model_key, self.params,
                                            expanded_o,
                                            jnp.expand_dims(term_tick, 1),
                                            h_tickminus1)
        critic_preds.append(v_tick)
        self.o_tick = o_tick.copy()
        self.r_tick = r_tick.copy()
        self.h_tickminus1 = jax.tree_map(lambda x: x, h_tickminus1)
        self.term_tick = term_tick.copy()
        hidden_stacked = jax.tree_map(lambda *args: jnp.stack(args, 1), *hiddens)
        
        # jax.debug.print("hidden_stacked {}", jnp.stack(hidden_indices, 1))
        

        return Namespace(**{
            'observations': observations,
            'actions': jnp.stack(actions, 1),
            'rewards': jnp.stack(rewards, 1),
            'terminations': jnp.stack(terminations, 1),
            'infos': infos,
            'critic_preds': jnp.squeeze(jnp.stack(critic_preds, 1), axis=-1),
            'actor_preds': jnp.stack(actor_preds, 1),
            'hiddens': hidden_stacked,
            'hidden_indices': jnp.stack(hidden_indices, 1),
            'masked': jnp.stack(masked, 1)
        })
    
    def evaluate(self,random_key,eval_episodes):
        #Evaluate the agent for evaluation_steps
        #Create a single zero hidden state
        #Get the hidden state from the first actor
        o_tick,_=self.eval_env.reset()
        episode_lens=[]
        episode_avgreturns=[]
        rollouts=[]
        term_tick=jnp.zeros((1,1),dtype=bool)  #Initialize terminal state to False
        #Initialize zero hidden state at the start of each episode, shape is infered from the hidden state of the first environment
        h_tickminus1=jax.tree_map(lambda x:jnp.expand_dims(jnp.zeros(x[0].shape),0) ,self.h_tickminus1)
        # print("eps", eval_episodes)
        for i in tqdm.tqdm(range(eval_episodes)):
            done=False
            rewards=[]
            
            while not done:
                #Take a step in the environment
                random_key,model_key=jax.random.split(random_key)
                # for k in o_tick.keys():
                #     print("bef", k, o_tick[k].shape)
               
                expanded_o = self.expand_o_tick(o_tick, eval=True)
                expanded_o = self.expand_o_tick(expanded_o, eval=True)
                # for k in expanded_o.keys():
                #     print("aft", k, expanded_o[k].shape)
                
                act_logits,v_tick,htick=self.actor_critic_fn(model_key,self.params,expanded_o,term_tick,h_tickminus1)
                # if hasattr(self,'arg_max') and self.arg_max:
                #     acts_tick=jnp.argmax(act_logits,axis=-1)
                # else:
                #     if self.use_gumbel_sampling:
                #          # sample action: Gumbel-softmax trick
                #         # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
                #         u = jax.random.uniform(random_key, shape=act_logits.shape)
                #         acts_tick=jnp.argmax(act_logits - jnp.log(-jnp.log(u)), axis=-1).squeeze(axis=-1)
                #     else:
                #         acts_tick=jax.random.categorical(random_key,act_logits).squeeze(axis=-1)
                acts_tick = self.sampling_impl.sampling_differ(act_logits, random_key, expanded_o.get("mask", None))
                
                # print("the shapes", act_logits.shape, acts_tick.shape)
                o_tick,r_tick,term,trunc,info=self.eval_env.step(*jax_to_numpy(acts_tick))
                # print("o_tick", o_tick, "r_tick", r_tick, "term", term, "trunc", trunc, "info", info)
                o_tick,r_tick=numpy_to_jax(o_tick,r_tick)
                done=term or trunc
                term_tick=jnp.array([[done]],dtype=bool) #Carry forward the termination signal for the next timestep, shape expected by the actor_critic_fn is BXT
                rewards.append(r_tick)
                h_tickminus1=htick
            #Get the rollout frames
            # print("info", jnp.array(info["final_info"]["actions"]).shape) 
            # print("info", info["final_info"]["eval_scaled_diff"].shape) 
            # actions = jnp.array(info["final_info"]["actions"])
            # scaled_diff = jnp.array(info["final_info"]["eval_scaled_diff"])
            # rew = jnp.array(info["final_info"]["rewards"])
            
            
            # max_x = jnp.array(info["final_info"]["max_x"])
            # # print("max_cof", max_x.shape, max_y.shape)
            # print("max_cof", max_x.shape)
            # conc_max = jnp.concatenate([max_x, jnp.array([[0,0]])], axis=1)
            # # print("max_cof", conc_max.shape)
            
            # eval_rew = jnp.zeros((rew.shape[0] * actions.shape[1],1), dtype=jnp.float32)  # Create an array filled with zeros
            # eval_rew = eval_rew.at[jnp.arange(rew.shape[0]) * actions.shape[1],1].set(rew)
            # # print("rew", rew)
            # # print(eval_rew)
            # # print("scaled_diff", scaled_diff.shape, rew.shape, actions.shape)
         
            # actions = jnp.reshape(actions, (actions.shape[0] * actions.shape[1], actions.shape[2]))
            # scaled_diff = jnp.reshape(scaled_diff, (scaled_diff.shape[0] * scaled_diff.shape[1], 1))
            
            # # print("actions", actions.shape)
            # combined = jnp.concatenate([actions, scaled_diff, eval_rew], axis=1)
            # table = jnp.concatenate([conc_max, combined], axis=0)
            
            rollouts = None
            episode_lens.append(len(rewards))
            rewards=jnp.array(rewards,dtype=jnp.float32)
            avg_return=rlax.discounted_returns(rewards,self.gamma*jnp.ones_like(rewards),jnp.zeros_like(rewards)).mean()
            episode_avgreturns.append(avg_return)
        avg_episode_len=jnp.array(episode_lens).mean()
        avg_episode_return=jnp.array(episode_avgreturns).mean()
        return avg_episode_len,avg_episode_return,rollouts

