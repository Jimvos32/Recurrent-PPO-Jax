import jax
import jax.numpy as jnp
import optax
import rlax
import flax
import tqdm
import jax.numpy as jnp
import numpy as np
import time
import optax

from flax.training.train_state import TrainState
from src.models.actor_critic import *
from typing import Callable,Tuple
from src.agents.base_agent import BaseAgent



class PPOAgent(BaseAgent):

    def __init__(self,train_envs,eval_env,repr_model_fn:Callable,seq_model_fn:Tuple[Callable,Callable],
                        actor_fn:Callable,critic_fn:Callable,optimizer:optax.GradientTransformation,
                         num_steps=128, gamma=0.99, lr_schedule=optax.linear_schedule,
                        gae_lambda=0.95, num_minibatches=4, update_epochs=4, norm_adv=True,
                        clip_coef=0.1, ent_schedule=optax.Schedule, vf_coef=0.5, max_grad_norm=0.5,
                        target_kl=None,sequence_length=None, task_name=None) -> None:

        super(PPOAgent,self).__init__(train_envs=train_envs,eval_env=eval_env,rollout_len=num_steps,repr_model_fn=repr_model_fn,seq_model_fn=seq_model_fn,
                        actor_fn=actor_fn,critic_fn=critic_fn,use_gumbel_sampling=False,sequence_length=sequence_length, continious_sampling=False, single_dim=False, task_name=task_name)
        
        self.optimizer=optimizer
        self.num_envs = self.env.num_envs
        self.gamma = gamma
        self.lr_schedule = lr_schedule
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.ent_schedule = ent_schedule
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_tick=jnp.array(0)
        self.task=task_name
        
    
        @jax.jit
        def update_ppo(
            params,optimizer_state,random_key,
            data_batch,update_tick
        ):
            
            #Update lr
            
            optimizer_state[1].hyperparams['learning_rate']=self.lr_schedule(update_tick)
            
            
            Glambda_fn=jax.vmap(rlax.lambda_returns)
            observations,actions,rewards,terminations,critic_preds,actor_preds=data_batch['observations'],data_batch['actions'], \
                                            data_batch['rewards'],data_batch['terminations'],data_batch['critic_preds'],data_batch['actor_preds']
                                            
            gammas=self.gamma*(1-terminations)
            lambdas=self.gae_lambda*jnp.ones(self.num_envs)
            #Calculate Lamba for timesteps G_{tick} - G_{tick+rollout_len}
            #rewards, gammas, lambdas values at timesteps {tick+1} - {tick+rollout_len+1}
            Glambdas=Glambda_fn(rewards[:,1:],gammas[:,1:],
                              critic_preds[:,1:],lambdas)
            #Calculate the advantages using timesteps {tick} - {tick+rollout_len}
            advantages=Glambdas-critic_preds[:,:-1]
            #Calculate log probs shape (num_envs*rollout_len,num_actions)
            def gaussian_log_prob(actions, act_logits):
                if not self.continious_sampling and not self.use_gumbel_sampling:
                    action_dim = act_logits.shape[-1] // 2
                    means = act_logits[..., :action_dim].squeeze()
                    log_stds = act_logits[..., action_dim:].squeeze()
                    
                    
                    # Clip log_stds for numerical stability
                    log_stds = jnp.clip(log_stds, -20.0, 2.0)
                    
                    
                    
                    variance = jnp.exp(2 * log_stds)
                    log_prob = -0.5 * (
                        jnp.log(2 * jnp.pi)
                        + 2 * log_stds
                        + (actions - means) ** 2 / variance
                    )
                    
                    
                else :    
                
                    action_dim = act_logits.shape[-1] // 2

                    # Extract means and log_stds
                    means = act_logits[..., :action_dim]  # Shape: (batch_size, action_dim)
                    log_stds = act_logits[..., action_dim:]  # Shape: (batch_size, action_dim)

                    # Clip log_stds for numerical stability
                    log_stds = jnp.clip(log_stds, -20.0, 2.0)
                    # print("outputs", act_logits.shape, "means", means.shape, "std", log_stds.shape, "split", action_dim)

                    # Expand means and log_stds to match action sampling dimensions (batch_size, num_samples, action_dim)
                    batch_dim = self.eval_env.unwrapped.batch_size
                    means = jnp.repeat(means, batch_dim, axis=2) # Shape: (8, 256, 3, 1)
                    means = jnp.reshape(means, means.shape[:3] + (-1,))
                    log_stds = jnp.expand_dims(jnp.repeat(log_stds, batch_dim, axis=2), axis=-1)  # Shape: (8, 256, 3, 1)
                    log_stds = jnp.reshape(log_stds, log_stds.shape[:3] + (-1,))
                
                    variance = jnp.exp(2 * log_stds)  # (batch_size, num_samples, action_dim)

                    # Compute log probability
                    log_prob = -0.5 * (
                        jnp.log(2 * jnp.pi) + 2 * log_stds + (actions - means) ** 2 / variance
                    )  # Shape: (batch_size, num_samples, action_dim)
                    
                    print("ear", log_prob)
                
                    log_prob = log_prob.sum(axis=2)  
                    
                print("pol_out", act_logits.shape, "means ", means.shape, "std ", log_stds.shape,"actions ", actions.shape, "logstds", log_prob.shape)
                return log_prob
            
            # def gaussian_log_prob(actions, act_logits):
                
                
            #     action_dim = act_logits.shape[-1] // 2
            #     means = act_logits[..., :action_dim].squeeze()
            #     log_stds = act_logits[..., action_dim:].squeeze()
                
                
            #     # Clip log_stds for numerical stability
            #     log_stds = jnp.clip(log_stds, -20.0, 2.0)
                
                
                
            #     variance = jnp.exp(2 * log_stds)
            #     log_prob = -0.5 * (
            #         jnp.log(2 * jnp.pi)
            #         + 2 * log_stds
            #         + (actions - means) ** 2 / variance
            #     )
                
            #     print("pol_out", act_logits.shape, "means ", means.shape, "std ", log_stds.shape,"actions ", actions.shape, "logstds", log_prob.shape)
                
            #     return log_prob

            logprobs = gaussian_log_prob(actions, actor_preds)
            
            # print("whats he logging", logprobs.shape)
            # print("probably", logprobs.shape)
            # B,T=actions.shape
            
            # # print("actions", actions.shape,actor_preds.shape)
            # logprobs=jax.nn.log_softmax(actor_preds).reshape(B*T,-1)
            # logprobs=logprobs[jnp.arange(B*T),actions.reshape(-1)].reshape(B,T)
            # #Calculate log probs of actions takenß
            
            
            def ppo_loss(params, random_key, mb_observations, mb_actions,mb_terminations,
                            mb_logp, mb_advantages, mb_returns,mb_h_tickminus1):
                logits_new,values_new,_=self.actor_critic_fn(random_key,params,mb_observations,mb_terminations,
                                                             mb_h_tickminus1)
                #newlogprob, entropy, newvalue = get_action_and_value2(random_key,params, x, a)
                # B,T=mb_actions.shape
                # newlogprobs=jax.nn.log_softmax(logits_new).reshape(B*T,-1)
                # newlogprobs=newlogprobs[jnp.arange(B*T),mb_actions.reshape(-1)].reshape(B,T)
                newlogprobs = gaussian_log_prob(mb_actions, logits_new)
                # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
                
                
                logits_new = logits_new - jax.scipy.special.logsumexp(logits_new, axis=-1, keepdims=True)
                logits_new = logits_new.clip(min=jnp.finfo(logits_new.dtype).min)
                p_log_p = logits_new * jax.nn.softmax(logits_new)
                entropy = -p_log_p.sum(-1)
                
                # print("logits", logits_new.shape, "logp", newlogprobs.shape, "entropy", entropy.shape, "values", mb_logp.shape)

                logratio = newlogprobs - mb_logp
                ratio = jnp.exp(logratio)
                approx_kl = ((ratio - 1) - logratio).mean()

                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                print("logits", logits_new.shape, "logp", newlogprobs.shape, "entropy", entropy.shape, "mb", mb_logp.shape, "ratio", ratio.shape, "adv", mb_advantages.shape, "returns", mb_returns.shape)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((values_new - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_schedule(update_tick) * entropy_loss + v_loss * self.vf_coef
                return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

            ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)


            #Use observations tick to tick+rollout_len
            observations=observations[:,:-1]
            terminations=terminations[:,:-1]
            hiddens=data_batch['hiddens'] #A Pytree of with the leading dimension of shape num_envs*num_seqs
            hidden_indices=data_batch['hidden_indices'] #A jax array of shape (num_envsXnum_seqsXseq_len)
            num_seqs=hidden_indices.shape[1]

            #We are gonna minibatch over num_envs and num_seqs now

            def update_epoch(carry,x):
                params,optimizer_state,random_key=carry
                shuffle_key,model_key,random_key = jax.random.split(random_key,3)
                shuffled_inds = jax.random.permutation(shuffle_key, self.num_envs*num_seqs)
                batch_inds = shuffled_inds.reshape((self.num_minibatches, -1))
                #We are gonna minibatch over num_envs and num_seqs now
                def minibatch_update(carry,x):
                    params,optimizer_state,model_key=carry
                    batch_ind=x
                    mbenvinds=batch_ind//num_seqs
                    mbseqinds=batch_ind%num_seqs
                    model_key, _ = jax.random.split(model_key)
                    hidden_indices_mb=hidden_indices[mbenvinds,mbseqinds]
                    mb_h_tickminus1=jax.tree_map(lambda x:x[mbenvinds,mbseqinds],hiddens)

                    #We first index by the minibatch envs and then by the indices for the corresponding timestep in those minibatches
                    # The first index is the env id and the second index is the timestep id
                    # In numpy the 2nd index needs column id corresponding to each row (ie each env id) so index with an array of shape (num_timesteps,num_envs)
                    # We need to transpose two times to get the right shape
                    mb_observations=observations[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,observations.ndim)))
                    mb_actions=actions[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,actions.ndim)))
                    mb_terminations=terminations[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,terminations.ndim)))
                    mb_logp=logprobs[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,logprobs.ndim)))
                    mb_advantages=advantages[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,advantages.ndim)))
                    mb_returns=Glambdas[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,Glambdas.ndim)))
                    (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                         params,
                         model_key,
                         mb_observations,
                         mb_actions,
                         mb_terminations,
                         mb_logp,
                         mb_advantages,
                         mb_returns,
                         mb_h_tickminus1
                     )
                    updates,optimizer_state = self.optimizer.update(grads, optimizer_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params,optimizer_state,model_key),(loss, pg_loss, v_loss, entropy_loss, approx_kl)
                
                (params,optimizer_state,model_key),losses=jax.lax.scan(minibatch_update,(params,optimizer_state,model_key),batch_inds)
                losses=jax.tree_map(lambda x:x.mean(),losses)
                return (params,optimizer_state,random_key),losses
            
            
            (params,optimizer_state,random_key),losses=jax.lax.scan(update_epoch,(params,optimizer_state,random_key),jnp.arange(self.update_epochs))
            losses=jax.tree_map(lambda x:x.mean(),losses)
            loss, pg_loss, v_loss, entropy_loss, approx_kl=losses
            return (loss, pg_loss, v_loss, entropy_loss, approx_kl),params, optimizer_state
        self.update_ppo = update_ppo

        
    def reset(self,params_key,random_key):
        super(PPOAgent,self).reset(params_key,random_key)
        self.optimizer_state=self.optimizer.init(self.params)
        self.update_tick=jnp.array(0)

    def step(self,random_key):
        #Unroll actor for rollout_len steps

        h_tickminus1=jax.tree_map(lambda x: x,self.h_tickminus1) #Copy the hidden states
        #Need to remove values
        unroll_key,update_key=jax.random.split(random_key)
        databatch=self.unroll_actors(unroll_key)
        databatch=vars(databatch)
        infos=databatch.pop('infos')
        (loss, pg_loss, v_loss, entropy_loss, approx_kl),self.params, self.optimizer_state=self.update_ppo(self.params,
                                self.optimizer_state,update_key,databatch,self.update_tick)
        
        rewards=databatch['rewards']
        self.update_tick=self.update_tick+1
        return (loss,(v_loss,entropy_loss,pg_loss,rewards),infos) #Will clean this up later




