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
import optax
import jax
import jax.numpy as jnp
from src.agents.ppo_dic_inherits.root_agent import RootAgent

class BasePPO(RootAgent):
    """
    PPOAgent builds on the generic agent logic in SamplingParent.
    It adds PPO update logic (not fully implemented here) and is configured
    with a task-specific sampling implementation.
    """
    def __init__(self,train_envs,eval_env,repr_model_fn:Callable,seq_model_fn:Tuple[Callable,Callable],
                        actor_fn:Callable,critic_fn:Callable,optimizer:optax.GradientTransformation, sampling_impl:type,
                        num_steps=128, gamma=0.99, lr_schedule=optax.linear_schedule,
                        gae_lambda=0.95, num_minibatches=4, update_epochs=4, norm_adv=True,
                        clip_coef=0.1, ent_schedule=optax.Schedule, stability_schedule=optax.Schedule, vf_coef=0.5, max_grad_norm=0.5, kl_coeff=0.1,
                        target_kl=None,sequence_length=None, sample_dist=1, task_name=None) -> None:
        
        self.sample_distribution=sample_dist
        super(BasePPO,self).__init__(train_envs=train_envs,eval_env=eval_env,rollout_len=num_steps,repr_model_fn=repr_model_fn,seq_model_fn=seq_model_fn,
                        actor_fn=actor_fn,critic_fn=critic_fn,sampling_impl_class=sampling_impl,sequence_length=None, single_dim=False, task_name=task_name)
        
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
        self.stability_schedule = stability_schedule
        self.vf_coef = vf_coef
        self.kl_coeff = kl_coeff
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_tick=jnp.array(0)
        self.task=task_name
        
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
        
        
    
        @jax.jit
        def update_ppo(
            params,optimizer_state,random_key,
            data_batch,update_tick
        ):
            
            #Update lr
            
            optimizer_state[1].hyperparams['learning_rate']=self.lr_schedule(update_tick)
            
            
            Glambda_fn=jax.vmap(rlax.lambda_returns)
            observations,actions,rewards,terminations,critic_preds,actor_preds, masked=data_batch['observations'],data_batch['actions'], \
                                            data_batch['rewards'],data_batch['terminations'],data_batch['critic_preds'],data_batch['actor_preds'], data_batch['masked']
                                            
                            
            gammas=self.gamma*(1-terminations)
            lambdas=self.gae_lambda*jnp.ones(self.num_envs)
            #Calculate Lamba for timesteps G_{tick} - G_{tick+rollout_len}
            #rewards, gammas, lambdas values at timesteps {tick+1} - {tick+rollout_len+1}
            # print("rewards", rewards.shape, "gammas", gammas.shape, "lambdas", lambdas.shape, "critic_preds", critic_preds.shape, "actor_preds", actor_preds.shape)
            
            Glambdas=Glambda_fn(rewards[:,1:],gammas[:,1:],
                              critic_preds[:,1:],lambdas)
            # print("Glambdas", Glambdas.shape, "critic_preds", rewards[:,1:].shape, "critix_preds", critic_preds[:,1:].shape)
            # jax.debug.print("terminations {} {}", (1 - terminations), rewards)
            #Calculate the advantages using timesteps {tick} - {tick+rollout_len}
            advantages=Glambdas-critic_preds[:,:-1]
            
            # scatter_dict = {}
            # s_act = actions.flatten()
            # s_rew = rewards[:,1:].flatten()
            # s_crit = critic_preds[:,1:].flatten()
            # s_adv = advantages.flatten()
            # s_gl = Glambdas.flatten()
            
            # scatter_dict['actions'] = s_act
            # scatter_dict['rewards'] = s_rew
            # scatter_dict['critic_preds'] = s_crit
            # scatter_dict['advantages'] = s_adv
            # scatter_dict['glambdas'] = s_gl
            
            
            
            # print("advantages", advantages.shape, "Glambdas", Glambdas.shape, "critic_preds", critic_preds.shape, "actor_preds", actions.shape, "rewards", rewards.shape)
            # jax.debug.print("glambdas \n{}\ncritic \n{}\rewards \n{}\nactions \n{}\n", Glambdas[:,0], critic_preds[:,0], rewards[:,0], actions[:,0])
            #Calculate log probs shape (num_envs*rollout_len,num_actions)
            logprobs=self.sampling_impl.gaussian_log_prob(actions,actor_preds)
            
            stability_coef = self.stability_schedule(update_tick)
            
            
      
            def ppo_loss(params, random_key, mb_observations, mb_actions, mb_masked, mb_terminations,
                            mb_logp, mb_advantages, mb_returns,mb_h_tickminus1):
                key, random_key = jax.random.split(random_key, 2)
                logits_new,values_new,_=self.actor_critic_fn(random_key,params,mb_observations,mb_terminations,
                                                             mb_h_tickminus1)
                
                
                # print("are we iterating", logits_new.shape, "old_log", mb_logp.shape, "val", values_new.shape, "obs_in", mb_observations["actions"].shape, mb_terminations.shape)
                average_logits = jnp.mean(logits_new, axis=(0, 1))

                newlogprobs = self.sampling_impl.gaussian_log_prob(mb_actions, logits_new)
                
                # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
                # print("logits", mb_logp.shape)
            
                entropy = self.sampling_impl.entropy(logits_new, mb_masked, key=key)
                # # print("entropyfg", entropy.shape)
                entropy = entropy.mean()
                # print("entropy", entropy.shape)
                
                max_clip = 20.0
                logratio = jnp.clip(newlogprobs - mb_logp, -max_clip, max_clip) 
                
                ratio = jnp.exp(logratio) # log
                approx_kl = ((ratio - 1) - logratio).mean()

                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((values_new - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                
                m_new_logprobs = jnp.mean(newlogprobs)
                m_logprobs = jnp.mean(mb_logp)
                m_ratio = jnp.mean(ratio)
                m_returns = jnp.mean(mb_returns)
                m_values = jnp.mean(values_new)
                mb_advantages = jnp.mean(mb_advantages)
                
                
                
                # jax.debug.print("l2_norm {}", l2_norm(params))
                # decay_coef = 0.01
                # weight_decay = l2_norm(params) * decay_coef
                # split = average_logits.shape[0] // 2
                # stds = average_logits[split:]
                
                
                
                # avg_stds = jnp.mean(stds)
                
                
                
                stable_pen = 1#(avg_stds  * 0 + l2_norm(params) * 0.2) * stability_coef
             
                # jax.debug.print("stable pen {} foed {} tick {} pg {} entrop {} v {}", stable_pen, stability_coef, update_tick, pg_loss, self.ent_schedule(update_tick) * entropy_loss, self.vf_coef *v_loss)
                
                # loss = pg_loss + v_loss * self.vf_coef #+ stable_pen
                loss = pg_loss - self.ent_schedule(update_tick) * entropy_loss + v_loss * self.vf_coef
                # jax.debug.print("extropy {}", entropy_loss)
                # loss = entropy_loss #* -1
                
                # jax.debug.print("pg {} {} {} ", v_loss, pg_loss, entropy_loss)
                # loss = entropy_loss * -1
                # print("loss", loss, "pg_loss", pg_loss, "v_loss", v_loss, "entropy_loss", entropy_loss, "s", s, "entropy", entropy)
                return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl), 
                              (m_new_logprobs, m_logprobs, m_ratio, m_returns, m_values, mb_advantages, average_logits, stable_pen, 1,1))

                    
            


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
                # jax.debug.print("opt {}", optimizer_state[1].hyperparams['learning_rate'])
                shuffle_key,model_key,random_key = jax.random.split(random_key,3)
                shuffled_inds = jax.random.permutation(shuffle_key, self.num_envs*num_seqs)
                batch_inds = shuffled_inds.reshape((self.num_minibatches, -1))
                # print("batch_inds", batch_inds.shape, "num_envs", self.num_envs, "num_seqs", num_seqs)
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
                    
                    mb_actions=actions[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,actions.ndim)))
                    mb_masked=masked[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,masked.ndim)))
                 
                    # print("mb_observations", mb_actions.shape, actions.shape, )
                    mb_terminations=terminations[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,terminations.ndim)))
                    mb_logp=logprobs[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,logprobs.ndim)))
                    mb_advantages=advantages[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,advantages.ndim)))
                    mb_returns=Glambdas[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,Glambdas.ndim)))
                    (loss, (pg_loss, v_loss, entropy_loss, approx_kl, update_info)), grads = ppo_loss_grad_fn(
                         params,
                         model_key,
                         mb_observations,
                         mb_actions,
                         mb_masked,
                         mb_terminations,
                         mb_logp,
                         mb_advantages,
                         mb_returns,
                         mb_h_tickminus1,
                     )
                    
                    # print("what is this", mbenvinds,hidden_indices_mb.T, actions.shape, mb_actions.shape) #(4, 6, 4, 3) (1, 3, 4, 3)
                    # jax.debug.print("ind {}, other {}", mbenvinds, hidden_indices_mb.T)
                    # jax.debug.print("actions\n{}\nmb_actions {} \n", actions, mb_actions)
                    
                    
                    
                    # jax.debug.print("actions: \n{}\mb_actions:\n", actions, mb_actions)
                    
                    updates,optimizer_state = self.optimizer.update(grads, optimizer_state, params)
                    
                    
                    
                    
                    params = optax.apply_updates(params, updates)
                    update_info = update_info + (l2_norm(grads),)
                    return (params,optimizer_state,model_key),(loss, pg_loss, v_loss, entropy_loss, approx_kl, update_info)
                #grads and params l2 norm
                (params,optimizer_state,model_key),losses=jax.lax.scan(minibatch_update,(params,optimizer_state,model_key),batch_inds)
              
                losses=jax.tree_map(lambda x:jnp.mean(x, axis=0),losses)
          
                # update_info=jax.tree_map(lambda x:x.mean(),update_info)
                return (params,optimizer_state,random_key),losses #,update_info
            
            
            (params,optimizer_state,random_key),losses=jax.lax.scan(update_epoch,(params,optimizer_state,random_key),jnp.arange(self.update_epochs))
            losses=jax.tree_map(lambda x:jnp.mean(x, axis=0),losses)
            # update_info=jax.tree_map(lambda x:x.mean(),update_info)
            loss, pg_loss, v_loss, entropy_loss, approx_kl, update_info = losses
            # update_info = update_info + (scatter_dict,)
            return (loss, pg_loss, v_loss, entropy_loss, approx_kl, update_info),params, optimizer_state#, update_info
        self.update_ppo = update_ppo

        
    def reset(self,params_key,random_key):
        super(BasePPO,self).reset(params_key,random_key)
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
        (loss, pg_loss, v_loss, entropy_loss, approx_kl, update_info),self.params, self.optimizer_state =self.update_ppo(self.params,
                                self.optimizer_state,update_key,databatch,self.update_tick)
        
        params_l2 = l2_norm(self.params)
        update_info = update_info + (params_l2,)
        # print("okay", self.params.shape)
        rewards=databatch['rewards']
        self.update_tick=self.update_tick+1
        # for k in update_info:
        #    print(k, k.shape)
        return (loss,(v_loss,entropy_loss,pg_loss,rewards), update_info, infos) #Will clean this up later



def l2_norm(pytree):
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(pytree)]))
