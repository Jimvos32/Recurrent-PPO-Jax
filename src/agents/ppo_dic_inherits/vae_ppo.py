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
from argparse import Namespace


class VAEPPO(RootAgent):
    """
    PPOAgent builds on the generic agent logic in SamplingParent.
    It adds PPO update logic (not fully implemented here) and is configured
    with a task-specific sampling implementation.
    """
    def __init__(self,train_envs,eval_env,repr_model_fn:Callable,seq_model_fn:Tuple[Callable,Callable],
                        actor_fn:Callable,critic_fn:Callable,latent_fn:Callable,pred_fn:Callable,optimizer:optax.GradientTransformation, sampling_impl:type,
                        num_steps=128, gamma=0.99, lr_schedule=optax.linear_schedule,
                        gae_lambda=0.95, num_minibatches=4, update_epochs=4, norm_adv=True,
                        clip_coef=0.1, ent_schedule=optax.Schedule, stability_schedule=optax.Schedule, vf_coef=0.5, max_grad_norm=0.5, kl_coeff=0.1,
                        target_kl=None,sequence_length=None, sample_dist=1, task_name=None) -> None:
        
        self.sample_distribution=sample_dist
        super(VAEPPO,self).__init__(train_envs=train_envs,eval_env=eval_env,rollout_len=num_steps,repr_model_fn=repr_model_fn,seq_model_fn=seq_model_fn,
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
        self.latent_fn=latent_fn
        self.pred_fn=pred_fn
        
        # Build the actor-critic model
        self.ac_model = nn.vmap(ActorCriticVAEModel,
                                variable_axes={'params': None},
                                split_rngs={'params': False, 'vae_sample': True})(
                                    repr_model_fn, self.seq_fn, actor_fn, critic_fn, latent_fn, pred_fn)
        
        @jax.jit
        def actor_critic_fn(random_key, params, inputs, terminations, last_memory):
            # jax.debug.print("is this thei first {} {} ", terminations.shape, inputs[inputs])
            random_key, vae_sample_key = jax.random.split(random_key)
            
        
            if terminations.shape == ():
                terminations = jnp.expand_dims(jnp.expand_dims(terminations, 0), 0)
            
            act_logits, values, memory, latent, target = self.ac_model.apply(
                params, inputs, terminations, last_memory,
                rngs={'random': random_key, 'vae_sample': vae_sample_key})
            
           
            
            # jax.debug.print("act_logits {} values {} memory{}, inputs {} terminations {}", 
            #                 act_logits.shape, values.shape, memory[0][0].shape, inputs["step"].shape, terminations.shape)
            return act_logits, values, memory, latent, target

        self.actor_critic_fn = actor_critic_fn
        
        
    
        @jax.jit
        def update_ppo(
            params,optimizer_state,random_key,
            data_batch,update_tick
        ):
            
            #Update lr
            
            optimizer_state[1].hyperparams['learning_rate']=self.lr_schedule(update_tick)
            
            
            Glambda_fn=jax.vmap(rlax.lambda_returns)
            observations,actions,rewards,terminations,critic_preds,actor_preds, masked, latent_mu, latent_std, target= \
            data_batch['observations'],data_batch['actions'], data_batch['rewards'],data_batch['terminations'], \
                data_batch['critic_preds'],data_batch['actor_preds'], data_batch['masked'], data_batch['latents_mu'], \
                    data_batch['latents_std'], data_batch['targets']
                                            
                            
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
            
            # print("advantages", advantages.shape, "Glambdas", Glambdas.shape, "critic_preds", critic_preds.shape, "actor_preds", actor_preds.shape, "rewards", rewards.shape)
            
            #Calculate the predictive loss
            # print("targetting", target.shape, "obs", observations["step"].shape, observations["observations"].shape)
            
            # jax.debug.print("target {} obs {}", target[0,:3], observations["observations"][0,:3])
            
            obs = observations["observations"][:, 1:]
            obs_target = target.transpose(0,1,3,2)
            recon_loss = jnp.square(obs - obs_target)
            
            def mask_fn(x, mask):
                N, S, B, _ = x.shape

                # Create a broadcastable index grid along B
                b_range = jnp.arange(B).reshape(1, 1, B, 1)  # shape [1, 1, B, 1]

                # Create boolean mask
                valid_mask = b_range < mask  # shape [N, S, B, 1]

                # Mask the data
                masked_data = x * valid_mask

                # Count valid entries along B
                valid_counts = valid_mask.sum(axis=2)  # shape [N, S, 1]

                # Compute average
                averaged = masked_data.sum(axis=2) / valid_counts
                
                return averaged
            
            masked_recon = mask_fn(recon_loss, masked)
            
            
            
            
            
            
            
            print("recon loss", recon_loss.shape, "obs", obs.shape, "target", obs_target.shape, masked.shape, "masked", masked.shape)
            
            # jax.debug.print("recon loss {} obs {} tar {} masked {}", recon_loss[0,:3], obs[0,:3], target[0,:3], masked_recon[0,:3])
            
            
            
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
                            mb_logp, mb_advantages, mb_returns,mb_h_tickminus1, mb_latent_mu, mb_latent_std, mb_target):
                key, random_key = jax.random.split(random_key, 2)
                logits_new,values_new,_, latent, target =self.actor_critic_fn(random_key,params,mb_observations,mb_terminations,
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
                # print(mb_actions.shape, mb_latent_mu.shape, mb_latent_std.shape, mb_target.shape)
                
                
                
                
                # avg_stds = jnp.mean(stds)
                
                
              
                
             
                # jax.debug.print("stable pen {} foed {} tick {} pg {} entrop {} v {}", stable_pen, stability_coef, update_tick, pg_loss, self.ent_schedule(update_tick) * entropy_loss, self.vf_coef *v_loss)
                def vae_loss(mb_latent_mu, mb_latent_std, mb_recon_losses):
                    kl_loss = -0.5 * jnp.sum(1 + mb_latent_std - jnp.square(mb_latent_mu) - jnp.exp(mb_latent_std), axis=-1).mean()
                    reconstruction_loss = jnp.sum(mb_recon_losses, axis=-1).mean()
                    return kl_loss, reconstruction_loss
                    
                kl_loss, reconstruct_loss = vae_loss(mb_latent_mu, mb_latent_std, masked_recon)
                variational_loss = kl_loss + reconstruct_loss * 0.01
                
                # print("kl_loss", kl_loss, "recon_loss", reconstruct_loss, "vae_loss", variational_loss, "entropy", entropy, "entropy_loss", entropy_loss)
                # jax.debug.print("kl_loss {} recon_loss {} vae_loss {} entropy {} entropy_loss {}", kl_loss, reconstruct_loss, variational_loss, entropy, entropy_loss)
                # loss = pg_loss + v_loss * self.vf_coef #+ stable_pen
                loss = pg_loss - self.ent_schedule(update_tick) * entropy_loss + v_loss * self.vf_coef + self.kl_coeff * variational_loss
                # jax.debug.print("extropy {}", entropy_loss)
                # loss = entropy_loss #* -1
                
                # print("loss", loss, "pg_loss", pg_loss, "v_loss", v_loss, "entropy_loss", entropy_loss, "s", s, "entropy", entropy)
                return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl), 
                              (m_new_logprobs, m_logprobs, m_ratio, m_returns, m_values, mb_advantages, average_logits, variational_loss, kl_loss, reconstruct_loss))

                    
            


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
                    
                    mb_latent_mu=latent_mu[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,latent_mu.ndim)))
                    mb_latent_std=latent_std[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,latent_std.ndim)))
                    mb_recon=masked_recon[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,masked_recon.ndim)))
                    # mb_target=target[mbenvinds,hidden_indices_mb.T].transpose((1,0)+tuple(range(2,target.ndim)))
                    
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
                         mb_latent_mu,
                         mb_latent_std,
                         mb_recon,
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
            update_info = update_info 
            return (loss, pg_loss, v_loss, entropy_loss, approx_kl, update_info),params, optimizer_state#, update_info
        self.update_ppo = update_ppo

        
    def reset(self,params_key,random_key):
        super(VAEPPO,self).reset(params_key,random_key)
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
        latents_mus = []
        latents_stds = []
        targets = []
        
        
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
            act_logits, v_tick, htick, latent, target = self.actor_critic_fn(model_key, self.params,
                                                               expanded_o,
                                                               jnp.expand_dims(term_tick, 1),
                                                               h_tickminus1)
            masks = expanded_o.get("mask", None)
           
            acts_tick = self.sampling_impl.sampling_differ(act_logits, random_key, masks)
            o_tickplus1, r_tickplus1, term_tickplus1, trunc_tickplus1, info = self.env.step(*self.jax_to_numpy(acts_tick))
            o_tickplus1, r_tickplus1 = self.numpy_to_jax(o_tickplus1, r_tickplus1)
            term_tickplus1, trunc_tickplus1 = self.numpy_to_jax(term_tickplus1, trunc_tickplus1, dtype=bool)
            term_tickplus1 = jnp.logical_or(term_tickplus1, trunc_tickplus1)
            critic_preds.append(v_tick.copy())
            actor_preds.append(act_logits.copy())
            actions.append(acts_tick.copy())
            latents_mus.append(latent[0].copy())
            latents_stds.append(latent[1].copy())
            targets.append(target.copy())
            infos.append(info)
            masked.append(masks)
            
            # print(jnp.array(actions).shape, "latent", latents[0].shape, "target", targets.shape, "laten std", latents[1].shape)
            
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
        _, v_tick, _, latent, target = self.actor_critic_fn(model_key, self.params,
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
        
        # print("actions", jnp.array(actions).shape, "loatent", jnp.array(latents_mus).shape, "target", jnp.array(targets).shape, "laten std", jnp.array(latents_stds).shape, observations["observations"].shape)
        

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
            'masked': jnp.stack(masked, 1),
            'latents_mu': jnp.stack(latents_mus, 1),
            'latents_std': jnp.stack(latents_stds, 1),
            'targets': jnp.stack(targets, 1),
        })



def l2_norm(pytree):
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(pytree)]))
