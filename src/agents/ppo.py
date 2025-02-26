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
from src.agents.base_agent_dic import BaseAgentDic



class PPOAgent(BaseAgentDic):

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
                                            
            # print("further ", actor_preds)
                                            
            gammas=self.gamma*(1-terminations)
            lambdas=self.gae_lambda*jnp.ones(self.num_envs)
            #Calculate Lamba for timesteps G_{tick} - G_{tick+rollout_len}
            #rewards, gammas, lambdas values at timesteps {tick+1} - {tick+rollout_len+1}
            Glambdas=Glambda_fn(rewards[:,1:],gammas[:,1:],
                              critic_preds[:,1:],lambdas)
            #Calculate the advantages using timesteps {tick} - {tick+rollout_len}
            advantages=Glambdas-critic_preds[:,:-1]
            #Calculate log probs shape (num_envs*rollout_len,num_actions)
            def gaussian_log_prob(task, actions, act_logits):
                # print("act_logits", act_logits.shape, "actions", actions.shape)
                if task == "sampling":# or self.task == "batch":
                    # print("action", actions.shape)
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
                    # print("log_prob", log_prob.shape, "means", means.shape, "stds", log_stds.shape, "actions", actions.shape)
                    
                    
                elif task == "batch":   
                    act_logits = jnp.reshape(act_logits, (actions.shape[0], act_logits.shape[1], 1, act_logits.shape[-1]))
                    actions = jnp.expand_dims(jnp.expand_dims(actions, axis=-1), axis=-1)
                    
                    action_dim = act_logits.shape[-1] // 2
                    batch_size = self.eval_env.unwrapped.batch_size
                    act_logits = jnp.repeat(act_logits, batch_size, axis=2)
                    means = act_logits[..., :action_dim]
                    log_stds = act_logits[..., action_dim:]
                    
                    
                
                    log_stds = jnp.clip(log_stds, -20.0, 2.0)
                    variance = jnp.exp(2 * log_stds)
                    # print("why ar ewe in here ", task)
                    log_prob = -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_stds + ((actions - means) ** 2) / variance)
                    # print("means", means.shape, "logstds", log_stds.shape, "actions", actions.shape, "act_logits", act_logits.shape, "action_dim", action_dim, "log_prob", log_prob.shape)
                    log_prob = jnp.squeeze(log_prob.sum(axis=2), axis=-1)
                    
                elif task == "multibatch":  
                   
                    act_logits = jnp.reshape(act_logits, (actions.shape[0], act_logits.shape[1], 1, act_logits.shape[-1])) # policy netywork output
                    #shaped [1,steps, 1, 2*action_dim] 
                    
                    
                    action_dim = act_logits.shape[-1] // 2
                    batch_size = self.eval_env.unwrapped.batch_size # amount of samples per step
                    means = act_logits[..., :action_dim]
                    log_stds = act_logits[..., action_dim:]
                    
                    
                
                    log_stds = jnp.clip(log_stds, -20.0, 2.0)
                    variance = jnp.exp(2 * log_stds)
                    log_prob = -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_stds + ((actions - means) ** 2) / variance)
                    #logprob calculation
                    # print("means", means.shape, "logstds", log_stds.shape, "actions", actions.shape, "act_logits", act_logits.shape, "action_dim", action_dim, "log_prob", log_prob.shape)
                    log_prob = jnp.squeeze(log_prob.sum(axis=-1), axis=-1)
                    
                elif task == "expanded_samp":
                    # print("act_logits", act_logits.shape, "actions", actions.shape)
                    act_logits = jnp.reshape(act_logits, (actions.shape[0], act_logits.shape[1], 1, act_logits.shape[-1]))
                    N, T, _, total_dim = act_logits.shape
                    k = total_dim // 3  # number of mixture components
                    act_logits = act_logits.squeeze(2)  # now shape: (N, T, 3*k)

                    # Split into means, log_stds, and weight logits.
                    means, log_stds, weight_logits = jnp.split(act_logits, 3, axis=-1)  # each is (N, T, k)
                    # Compute the mixture weights from the logits.
                    weights = jax.nn.softmax(weight_logits, axis=-1)  # (N, T, k)

                    # Broadcast parameters so they can be applied to each action in the batch.
                    # actions shape: (N, T, batch_size, 1)
                    batch_size = actions.shape[2]
                    means = jnp.broadcast_to(means[:, :, None, :], (N, T, batch_size, k))
                    log_stds = jnp.broadcast_to(log_stds[:, :, None, :], (N, T, batch_size, k))
                    weights = jnp.broadcast_to(weights[:, :, None, :], (N, T, batch_size, k))

                    stds = jnp.exp(log_stds)

                    # Compute the log probability for each Gaussian component.
                    # Note: we assume actions is broadcastable to (N, T, batch_size, k).
                    gaussian_log_probs = -0.5 * (((actions - means) / stds) ** 2) \
                                        - log_stds \
                                        - 0.5 * jnp.log(2 * jnp.pi)

                    # Combine with the mixture weights using log-sum-exp.
                    # Add a small epsilon to weights before taking log to avoid numerical issues.
                    log_weights = jnp.log(weights + 1e-8)
                    # log_prob_individual shape: (N, T, batch_size)
                    log_prob_individual = jax.scipy.special.logsumexp(log_weights + gaussian_log_probs, axis=-1)

                    # For a joint log probability over the batch of actions in a step, sum the individual log probabilities.
                    # joint_log_prob shape: (N, T)
                    log_prob = jnp.sum(log_prob_individual, axis=2)
                    # print("l", log_prob.shape)
                    
                elif task == "multidim":
                    
                    act_logits = jnp.reshape(act_logits, (actions.shape[0], act_logits.shape[1], 1, act_logits.shape[-1]))
                    action_dim = self.eval_env.unwrapped.action_dim  # Extract action_dim from the last dimension
                    batch_size = self.eval_env.unwrapped.batch_size  # Amount of samples per step

                    # Reshape act_logits to align with actions shape (1, steps, 1, 2 * action_dim)
                    act_logits = jnp.reshape(act_logits, (actions.shape[0], act_logits.shape[1], 1, act_logits.shape[-1]))  
                    
                    # Split into means and log_stds
                    means, log_stds = jnp.split(act_logits, 2, axis=-1)  # Shape: (1, steps, 1, action_dim)
                    
                    # Clamp log_stds for numerical stability
                    log_stds = jnp.clip(log_stds, -20, 2)
                    stds = jnp.exp(log_stds)

                    # Broadcast means and stds across the batch size dimension
                    # New shape: (1, steps, batch_size, action_dim)
                    means = jnp.broadcast_to(means, (act_logits.shape[0], act_logits.shape[1], batch_size, action_dim))
                    stds = jnp.broadcast_to(stds, (act_logits.shape[0], act_logits.shape[1], batch_size, action_dim))
                    
                    # Compute the variance
                    variance = stds ** 2  

                    # Compute log probability per action dimension
                    log_prob_per_dim = -0.5 * (
                        jnp.log(2 * jnp.pi) + 2 * log_stds + ((actions - means) ** 2) / variance
                    )  # Shape: (1, steps, batch_size, action_dim)
                    
                    # Sum over action dimensions to get total log probability per action sample
                    log_prob = jnp.sum(log_prob_per_dim, axis=-1)  # Shape: (1, steps, batch_size)
                    log_prob = jnp.sum(log_prob, axis=-1)
                    # log_prob = jnp.squeeze(log_prob, axis=-1)  # Remove unnecessary singleton dimension
                    
                    # print("means", means.shape, "logstds", log_stds.shape, "actions", actions.shape, "act_logits", act_logits.shape, "log_prob", log_prob.shape, "fas", log_prob_per_dim)
                                    
                # print("pol_out", act_logits.shape, "means ", means.shape, "std ", log_stds.shape,"actions ", actions.shape, "logstds", log_prob.shape)
                return log_prob
            
         
            # print("hell ueah", self.task, actions.shape, actor_preds.shape)
            logprobs = gaussian_log_prob(self.task, actions, actor_preds)
            # logprobs = gaussian_log_prob("sampling", actions, actor_preds)
            # jax.debug.print("logprobs {} \nlog_2 {}", logprobs, logprobs_2)
            
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
                # print("actions", mb_actions.shape, "logits", logits_new.shape)
                newlogprobs = gaussian_log_prob(self.task, mb_actions, logits_new)
                # newlogprobs_2 = gaussian_log_prob("sampling", mb_actions, logits_new)
                # print("mb_actions", mb_actions.shape, "logits.shape", logits_new.shape)
                # jax.debug.print("logprobs {} \nlog_2 {}", newlogprobs, newlogprobs_2)
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
                    
                # print("logits", logits_new.shape, "logp", newlogprobs.shape, "entropy", entropy.shape, "mb", mb_logp.shape, "ratio", ratio.shape, "adv", mb_advantages.shape, "returns", mb_returns.shape)

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
           
            # print("observations", observations["actions"].shape)
            def remove_last_from_dict(x):
                return {k:v[:,:-1] for k,v in x.items()}
            observations = remove_last_from_dict(observations)
            # print("observations", observations["actions"].shape)
            
            # print("observations", observations.shape)
            #We need to remove the last timestep from the observations,actions,terminations and logprobs
            
            # observations=observations[:,:-1]
            # print("observations", observations.shape)
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
                    
                    def take_from_dict(dictio):
                        return {k:v[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,v.ndim))) for k,v in dictio.items()}
                    
                    mb_observations=take_from_dict(observations)
                    # for k in mb_observations:
                    #     print(k, mb_observations[k].shape)
                    # print(hidden_indices_mb)
                    # print("mben", mbenvinds.shape, "hidden", hidden_indices_mb.shape, (1,0)+tuple(range(2,observations.ndim)), observations[mbenvinds,hidden_indices_mb.T].shape)
                    
                    # mb_observations=observations[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,observations.ndim)))
                    # print("mb_observations", mb_observations.shape, observations.shape)
                    
                    mb_actions=actions[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,actions.ndim)))
                    # print("mb_observations", mb_actions.shape, actions.shape, )
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
        # print("databatch", databatch.actor_preds.shape)
        databatch=vars(databatch)
        # print("databatch2", databatch["actor_preds"].shape)
        infos=databatch.pop('infos')
        (loss, pg_loss, v_loss, entropy_loss, approx_kl),self.params, self.optimizer_state=self.update_ppo(self.params,
                                self.optimizer_state,update_key,databatch,self.update_tick)
        
        rewards=databatch['rewards']
        self.update_tick=self.update_tick+1
        return (loss,(v_loss,entropy_loss,pg_loss,rewards),infos) #Will clean this up later




