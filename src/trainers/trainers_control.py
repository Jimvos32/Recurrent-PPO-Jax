import json
import numpy as np
import optax
import pandas as pd
import rlax
import wandb
import gymnasium as gym
import time
import logging
from argparse import Namespace
from src.trainers.base_trainer import BaseTrainer
from collections import OrderedDict
from src.tasks.envs.minigrid_env import create_minigrid_env_onehot,create_minigrid_env_pixel, create_sampling_env, create_multi_dim_env, create_multi_batch_env, create_mbatch, create_multi_dim, create_ackley, create_cosine, create_poly
from src.agents.a2c import A2CAgent
from src.agents.ppo import PPOAgent
from src.agents.ppo_vae import PPOAgentVAE
from src.agents.ppo_norm import PPOAgentNorm
from src.model_fns import *
from src.tasks.envs.wrappers import *
from src.trainers.utils import *
from gymnasium.wrappers import AutoResetWrapper
from omegaconf import DictConfig, OmegaConf
from src.model_fns.norm_fns import planar_flow, autoregressive_flow
from src.agents.ppo_dic_inherits.basic_ppo import BasePPO
from src.agents.ppo_dic_inherits.vae_ppo import VAEPPO
from src.agents.ppo_dic_inherits.inh_agents.full_params_agent import FullParamsSampling
from src.agents.ppo_dic_inherits.inh_agents.cor_gmm_agent import CorrelatedGaussianMixture 
from src.agents.ppo_dic_inherits.inh_agents.low_mvn_agent import LowRankMVN 
from src.agents.ppo_dic_inherits.inh_agents.flow_jax_agent import FlowMVN




logger = logging.getLogger(__name__)


def create_train_eval_envs(env_config):
    max_value = max(env_config['batches'])
    env_config['max_batches'] = max_value
    b_split = len(env_config['batches']) // 2
    train_b, eval_b = env_config['batches'][:b_split], env_config['batches'][b_split:]
    env_config['batches'] = train_b
    eval_config = env_config.copy()
    eval_config['batches'] = eval_b
    
    
    if env_config['env'] == "polynominal":
        train_fn=lambda: create_poly(**env_config)        
        eval_fn=lambda: create_poly(**eval_config)
    
    elif env_config['env'] == "cosine":
        train_fn=lambda: create_cosine(**env_config)
        eval_fn=lambda: create_cosine(**eval_config)
    elif env_config['env'] == "ackley":
        train_fn=lambda: create_ackley(**env_config)
        eval_fn=lambda: create_ackley(**eval_config)
        
    repr_fn=dict_unpack_mask(batch_expand_hidden=env_config["batch_expand_hidden"], 
                                 batch_combine_hidden=env_config["batch_combine_hidden"], 
                                 step_expand_hidden=env_config["step_expand_hidden"], 
                                 input_combine_hidden=env_config["input_combine_hidden"])
    
    return train_fn,eval_fn, repr_fn


def get_env_initializers(env_config):
    env_config=OmegaConf.to_container(env_config)
    if env_config['task']=='minigrid_pixel':
        env_fn=lambda: create_minigrid_env_pixel(**env_config)
        repr_fn=atari_conv_repr_model()
        return env_fn,env_fn,repr_fn
    elif env_config['task']=='minigrid_onehot':
        env_fn=lambda: create_minigrid_env_onehot(**env_config)
        repr_fn=flatten_repr_model()
        return env_fn,env_fn,repr_fn
    elif env_config['task']=='sampling':
        env_fn=lambda: create_sampling_env(**env_config)
        repr_fn=flatten_repr_model()
        return env_fn,env_fn,repr_fn
    elif env_config['task']=='multi':
        env_fn=lambda: create_multi_dim_env(**env_config)
        repr_fn=mlp_repr_model()
        return env_fn,env_fn,repr_fn
    elif env_config['task']=='batch':
        # print("config", env_config)
        env_fn=lambda: create_multi_batch_env(**env_config)
        repr_fn=mlp_repr_model()
        return env_fn,env_fn,repr_fn
    elif env_config['task']=='multibatch':
        # print("config", env_config)
        env_fn=lambda: create_mbatch(**env_config)
        repr_fn=mlp_repr_model()
        return env_fn,env_fn,repr_fn
    elif env_config['task']=='expanded_samp':
        # print("config", env_config)
        env_fn=lambda: create_mbatch(**env_config)
        repr_fn=mlp_repr_model()
        return env_fn,env_fn,repr_fn
    elif env_config['task']=='multidim':
        # print("config", env_config)
        env_fn=lambda: create_multi_dim(**env_config)
        repr_fn=dict_unpack_model()
        return env_fn,env_fn,repr_fn
    elif env_config['task']=='masked':
        train_fn,eval_fn,repr_fn=create_train_eval_envs(env_config)
        return train_fn,eval_fn,repr_fn
    elif env_config['task']=='gen_gmm':
        train_fn,eval_fn,repr_fn=create_train_eval_envs(env_config)
        return train_fn,eval_fn,repr_fn
    elif env_config['task']=='cor_gmm':
        train_fn,eval_fn,repr_fn=create_train_eval_envs(env_config)
        return train_fn,eval_fn,repr_fn
    elif env_config['task']=='full_params':
        train_fn,eval_fn,repr_fn=create_train_eval_envs(env_config)
        return train_fn,eval_fn,repr_fn
    elif env_config['task']=='vae':
        train_fn,eval_fn,repr_fn=create_train_eval_envs(env_config)
        return train_fn,eval_fn,repr_fn
    elif env_config['task']=='low_mvn':
        train_fn,eval_fn,repr_fn=create_train_eval_envs(env_config)
        return train_fn,eval_fn,repr_fn
    elif env_config['task']=='flow_jax':
        train_fn,eval_fn,repr_fn=create_train_eval_envs(env_config)
        return train_fn,eval_fn,repr_fn
    
def get_flow_func(flow_model, action_dim):
    if (flow_model == "planar_flow"):
        return planar_flow(action_dim)
    elif (flow_model == "auto_reg"):
        return autoregressive_flow(action_dim)
    else:
        return None
        
       
        

class ControlTrainer(BaseTrainer):

    def __init__(self,**kwargs):
        """
            Initialize the MiniGridTrainer class.

            Args:
            - wandb_run: wandb run object.
            - trainer_config: A dictionary of training configuration parameters.
                - rollout_len: Length of a rollout.
                - num_actors: Number of actors to run in parallel.
                - gamma: Discount factor.
                - model: Model to use, either 'lstm' or 'repr_minigrid'.
                - d_model: Dimension of the model.
                - n_layers: Number of layers in the LSTM model.
                - d_actor: Dimension of the actor model.
                - d_critic: Dimension of the critic model.
                - lr: Learning rate.
                - max_grad_norm: Maximum gradient norm.
                - agent: Agent to use, either 'a2c' or 'ppo'.
                - lamb: Lambda value.
                - value_coef: Coefficient for the value loss.
                - entropy_coef: Coefficient for the entropy loss.
                - record_steps_interval: Steps interval for recording.
            - env_config: A dictionary of environment configuration parameters.
            - global_args: A dictionary of global configuration parameters.
            - key: Random key for Jax.
            - seed: Seed for random number generation.
        """
        env_fn,eval_env_fn,repr_fn=get_env_initializers(kwargs['env_config'])
        flow_fn = get_flow_func(kwargs['trainer_config']['dist_model'], kwargs['env_config']['action_dim'])
        self.wandb_run=kwargs['wandb_run']
        self.trainer_config=kwargs['trainer_config']
        self.env_config=kwargs['env_config']
        self.global_config=kwargs['global_args']
        self.checkpoint_dir=self.global_config.get('checkpoint_dir',None)
        self.checkpoint_interval=self.global_config.get('checkpoint_interval',None)
        self.rollout_len=self.trainer_config['rollout_len']
        self.num_envs=self.trainer_config['num_envs']
        self.gamma=self.trainer_config['gamma']
        #We will create two environments, one for training (vectorized) and one for evaluation
        train_seeds=np.random.randint(0,9999,size=self.num_envs,dtype=int).tolist()
        eval_seeds=int(np.random.randint(0,9999,size=1,dtype=int))
        env_type=kwargs['trainer_config'].get('env_pool','async')
        # print("env type", env_type)
        env_type = 'sync'
        if env_type=='async':
            
            env_type=gym.vector.AsyncVectorEnv
        elif env_type=='sync':
            env_type=gym.vector.SyncVectorEnv
        train_envs=env_type([lambda: EpisodeStatisticsWrapper(AutoResetWrapper((env_fn())))for seed in train_seeds])#,shared_memory=False)
       

        eval_env=RecordRollout(AutoResetWrapper(eval_env_fn()))
        train_envs.reset(seed=train_seeds)
        eval_env.reset(seed=eval_seeds)
        logger.info("Observation space: "+str(eval_env.observation_space))
        logger.info("Action space: "+str(eval_env.action_space))
        

        params_key,self.random_key=jax.random.split(kwargs['key'])
        
        if self.trainer_config.seq_model.name=='lstm':
            model_fn=seq_model_lstm(**self.trainer_config['seq_model'])
        elif self.trainer_config.seq_model.name=='gru':
            model_fn=seq_model_gru(**self.trainer_config['seq_model'])
        elif self.trainer_config.seq_model.name=='gtrxl':
            model_fn=seq_model_gtrxl(**self.trainer_config['seq_model'])
            
            
        name = eval_env.unwrapped.name
        vae = False
        model_dist = self.trainer_config.get('dist_model', "standard")
        
        if model_dist == "vae":
            vae = True
        # vae = self.trainer_config.get('vae', False)
            
        if isinstance(eval_env.action_space, gym.spaces.Discrete):
            actor_fn = actor_model_discete(self.trainer_config['d_actor'],eval_env.action_space.n)
        elif name == "sampling" or name == "batch":
            # print("environment action space", eval_env.action_space.shape)
            actor_fn = actor_model_continuous(self.trainer_config['d_actor'], eval_env.action_space.shape)
        elif name == "expanded_samp":
            # print ("we doing the weird side step")
            actor_fn = actor_model_gmm(self.trainer_config['d_actor'], self.trainer_config['sample_distribution'])
        elif name == "multidim":
            if vae:
                actor_fn = vae_action_head(eval_env.unwrapped.action_dim, seq_hidden_sizes=self.trainer_config['d_actor'],  
                                           policy_hidden_sizes=self.trainer_config['actor_params_hidden'], latent_dim=self.trainer_config['latent_dim'])
            else:
                actor_fn = standard_action_head(eval_env.unwrapped.action_dim, seq_hidden_sizes=self.trainer_config['d_actor'], 
                                                policy_layers=self.trainer_config['actor_params_hidden'])
            # actor_fn = actor_model_continuous(self.trainer_config['d_actor'], (eval_env.unwrapped.action_dim, 0))
        elif name == "masked":
            if vae:
                actor_fn = vae_action_head(eval_env.unwrapped.action_dim, seq_hidden_sizes=self.trainer_config['d_actor'],  
                                           policy_hidden_sizes=self.trainer_config['actor_params_hidden'], latent_dim=self.trainer_config['latent_dim'])
            else:
                actor_fn = standard_action_head(eval_env.unwrapped.action_dim, seq_hidden_sizes=self.trainer_config['d_actor'], 
                                                policy_layers=self.trainer_config['actor_params_hidden'])
            # actor_fn = actor_model_continuous_params(self.trainer_config['d_actor'], list(self.trainer_config['actor_params_hidden']) + [eval_env.unwrapped.action_dim])
        elif name == "gen_gmm":
            sampling_imp = CorrelatedGaussianMixture
            gmm_components = self.trainer_config['sample_distribution'] * eval_env.unwrapped.max_batches
            if vae:
                actor_fn = gmm_vae_action_head(gmm_components, gmm_components,
                                    shared_seq_sizes=self.trainer_config['d_actor'], policy_hidden_sizes=self.trainer_config['actor_params_hidden'],
                                    latent_dim=self.trainer_config['latent_dim'])
            else:
                actor_fn = gmm_action_head(gmm_components, gmm_components,
                                    shared_seq_sizes=self.trainer_config['d_actor'], policy_hidden_sizes=self.trainer_config['actor_params_hidden'])
                
        elif name == "low_mvn":
            sampling_imp = LowRankMVN
            flat_act = eval_env.unwrapped.action_dim * eval_env.unwrapped.max_batches
            print("flat_act", flat_act, "rank", self.trainer_config['sample_distribution'], "env", eval_env.unwrapped.action_dim, "max_batches", eval_env.unwrapped.max_batches)
            if vae:
                actor_fn = gmm_vae_action_head(flat_act, self.trainer_config['sampling_distribution'],
                                    shared_seq_sizes=self.trainer_config['d_actor'], policy_hidden_sizes=self.trainer_config['actor_params_hidden'],
                                    latent_dim=self.trainer_config['latent_dim'])
            else:
                print("ciiof", self.env_config)
                actor_fn = mvn_action_head(flat_act, self.trainer_config['sample_distribution'],
                                    shared_seq_sizes=self.trainer_config['d_actor'], policy_hidden_sizes=self.trainer_config['actor_params_hidden'])
                
            # actor_fn = actor_gmm_params(self.trainer_config['d_actor'], list(self.trainer_config['actor_params_hidden']) + 
            #                             [self.trainer_config['sample_distribution'] * eval_env.unwrapped.action_dim])
        elif name == "cor_gmm":
            sampling_imp = CorrelatedGaussianMixture
            
            if vae:
                actor_fn = gmm_vae_action_head(self.trainer_config['sample_distribution'], 
                                    self.trainer_config['sample_distribution'] * eval_env.unwrapped.action_dim * eval_env.unwrapped.max_batches,
                                    shared_seq_sizes=self.trainer_config['d_actor'], policy_hidden_sizes=self.trainer_config['actor_params_hidden'],
                                    latent_dim=self.trainer_config['latent_dim'])
            else:
                actor_fn = gmm_action_head(self.trainer_config['sample_distribution'], 
                                    self.trainer_config['sample_distribution'] * eval_env.unwrapped.action_dim * eval_env.unwrapped.max_batches,
                                    shared_seq_sizes=self.trainer_config['d_actor'], policy_hidden_sizes=self.trainer_config['actor_params_hidden'])
                
            # gmm_components = self.trainer_config['sample_distribution'] * eval_env.unwrapped.action_dim
            # print("gmm_components", gmm_components)
            # actor_fn = actor_correlated_gmm(self.trainer_config['d_actor'], self.trainer_config['sample_distribution'], 
            #                                 list(self.trainer_config['actor_params_hidden']) + 
            #                             [self.trainer_config['sample_distribution'] * eval_env.unwrapped.action_dim * eval_env.unwrapped.max_batches])
        elif name == "full_params":
            sampling_imp = FullParamsSampling

            # if vae:
            #     actor_fn = vae_action_head(eval_env.unwrapped.max_batches * eval_env.unwrapped.action_dim, seq_hidden_sizes=self.trainer_config['d_actor'],  
            #                                policy_hidden_sizes=self.trainer_config['actor_params_hidden'], latent_dim=self.trainer_config['latent_dim'])
            # else:
            self.outsize = eval_env.unwrapped.max_batches * eval_env.unwrapped.action_dim
            
            actor_fn = standard_action_head(eval_env.unwrapped.max_batches * eval_env.unwrapped.action_dim, seq_hidden_sizes=self.trainer_config['d_actor'], 
                                            policy_layers=self.trainer_config['actor_params_hidden'])

            # actor_fn = actor_full_params(self.trainer_config['d_actor'], list(self.trainer_config['actor_params_hidden']) + 
            #                             [ eval_env.unwrapped.max_batches * eval_env.unwrapped.action_dim])
        
        elif name == "vae":
            
            actor_fn = variational(self.trainer_config['d_actor'], list(self.trainer_config['actor_params_hidden']) + 
                                        [self.trainer_config['latent_dim']], self.trainer_config['decoder_params_hidden'] + [eval_env.unwrapped.action_dim])
            
        elif name == "flow_jax":
            sampling_imp = FlowMVN
            
            policy_out = eval_env.unwrapped.action_dim
            # policy_out = eval_env.unwrapped.action_dim * eval_env.unwrapped.max_batches
            # policy_out = covariance + covariance * (covariance + 1) // 2
            
            
            actor_fn = mvn_action_head(policy_out, self.trainer_config['sample_distribution'],
                                    shared_seq_sizes=self.trainer_config['d_actor'], policy_hidden_sizes=self.trainer_config['actor_params_hidden'])
            
        

        critic_fn=critic_model(self.trainer_config['d_critic'])
        #Setup optimizer
        
        if self.trainer_config['agent']=='a2c':
            self.optimizer=optax.chain(optax.clip_by_global_norm(self.trainer_config['max_grad_norm']),  # Clip by the gradient by the global norm.
                                    optax.adam(**self.trainer_config.optimizer))  # Use Adam optimizer with learning rate.
            self.agent=A2CAgent(train_envs=train_envs,eval_env=eval_env,optimizer=self.optimizer, repr_model_fn=repr_fn,
                                seq_model_fn=model_fn,actor_fn=actor_fn,critic_fn=critic_fn,
                                rollout_len=self.rollout_len,
                                gamma=self.trainer_config['gamma'],lamb=self.trainer_config['lamb'],
                                value_loss_coef=self.trainer_config['value_coef'],
                                entropy_coef=self.trainer_config['entropy_coef'],
                                arg_max=self.trainer_config['arg_max'])
        elif self.trainer_config['agent']=='ppo':
            #Used from CleanRL PPO implementation
            batch_size = self.trainer_config['num_envs']*self.trainer_config['rollout_len'] 
            num_updates = self.global_config.steps // batch_size
            optimizer_config = dict(self.trainer_config.optimizer)
            learning_rate=optimizer_config.pop("learning_rate")
            if learning_rate['final'] is None:
                learning_rate['final']=learning_rate['initial'] #Set to none if you don't want decay
            if self.trainer_config['ent_coef']['final'] is None:
                self.trainer_config['ent_coef']['final']=self.trainer_config['ent_coef']['initial']
            
            
            lr_schedule=optax.polynomial_schedule(learning_rate['initial'],learning_rate['final'],learning_rate['power'],learning_rate['max_decay_steps'])
            ent_schedule=optax.polynomial_schedule(self.trainer_config['ent_coef']['initial'],self.trainer_config['ent_coef']['final'],
                                                   self.trainer_config['ent_coef']['power'],self.trainer_config['ent_coef']['max_decay_steps'])
            stability_schedule=optax.polynomial_schedule(self.trainer_config['stability_coef']['initial'],self.trainer_config['stability_coef']['final'],
                                                   self.trainer_config['stability_coef']['power'],self.trainer_config['stability_coef']['max_decay_steps'])

            self.optimizer=optax.chain(
                                optax.clip_by_global_norm(self.trainer_config['max_grad_norm']),
                                optax.inject_hyperparams(optax.adamw)(
                                    learning_rate=self.trainer_config['ent_coef']['initial'], **optimizer_config
                                ),
                            )
            
            sequence_steps = eval_env.unwrapped.max_episode_steps
            print("se", sequence_steps)
            
            agent_config = {
                "train_envs": train_envs,
                "eval_env": eval_env,
                "optimizer": self.optimizer,
                "repr_model_fn": repr_fn,
                "seq_model_fn": model_fn,
                "actor_fn": actor_fn,
                "critic_fn": critic_fn,
                "lr_schedule": lr_schedule,
                "ent_schedule": ent_schedule,
                "stability_schedule": stability_schedule,
                

                "num_steps": self.rollout_len,
                "gamma": self.trainer_config.get("gamma", 0.99),
                "gae_lambda": self.trainer_config.get("gae_lambda", 0.95),
                "num_minibatches": self.trainer_config.get("num_minibatches", 4),
                "update_epochs": self.trainer_config.get("update_epochs", 4),
                "norm_adv": self.trainer_config.get("norm_adv", True),
                "clip_coef": self.trainer_config.get("clip_coef", 0.1),
                "vf_coef": self.trainer_config.get("vf_coef", 0.5),
                "max_grad_norm": self.trainer_config.get("max_grad_norm", 0.5),
                "target_kl": self.trainer_config.get("target_kl", None),
                "sequence_length": sequence_steps,#self.trainer_config.get("sequence_length", None),
                "sample_dist": self.trainer_config.get("sample_distribution", 1),
                "task_name": self.env_config.get("task", None),
                "sampling_impl": sampling_imp
            }

            
            if(self.trainer_config['dist_model'] == "vae"):
                
                   # if vae:
            #     actor_fn = vae_action_head(eval_env.unwrapped.max_batches * eval_env.unwrapped.action_dim, seq_hidden_sizes=self.trainer_config['d_actor'],  
            #                                policy_hidden_sizes=self.trainer_config['actor_params_hidden'], latent_dim=self.trainer_config['latent_dim'])
                
                agent_config['latent_fn'] = latent_model(shared_hidden_sizes=self.trainer_config['lstm_seqential'], latent_dim=self.trainer_config['latent_dim'])
                # agent_config['pred_fn'] = recon_head(recon_hidden_layer=self.trainer_config["recon_hidden"], out_size=self.outsize)#action reconstruction
                agent_config['pred_fn'] = recon_head(recon_hidden_layer=self.trainer_config["recon_hidden"], out_size=eval_env.unwrapped.max_batches)#observation reconstruction
                self.agent=VAEPPO(**agent_config)
                
            elif(self.trainer_config['dist_model'] == "planar_flow" or self.trainer_config['dist_model'] == "auto_reg"):
                self.agent=PPOAgentNorm(train_envs=train_envs,eval_env=eval_env,optimizer=self.optimizer, repr_model_fn=repr_fn,
                                    seq_model_fn=model_fn,actor_fn=actor_fn,critic_fn=critic_fn,norm_flow=flow_fn,
                                    num_steps=self.rollout_len,
                                    gamma=self.trainer_config.get('gamma', 0.99),
                                    gae_lambda=self.trainer_config.get('gae_lambda', 0.95),
                                    num_minibatches=self.trainer_config.get('num_minibatches', 4),
                                    update_epochs=self.trainer_config.get('update_epochs', 4),
                                    norm_adv=self.trainer_config.get('norm_adv', True),
                                    clip_coef=self.trainer_config.get('clip_coef', 0.1),
                                    lr_schedule=lr_schedule,
                                    ent_schedule=ent_schedule,
                                    vf_coef=self.trainer_config.get('vf_coef', 0.5),
                                    max_grad_norm=self.trainer_config.get('max_grad_norm', 0.5),
                                    target_kl=self.trainer_config.get('target_kl', None),
                                    sequence_length=self.trainer_config.get('sequence_length', None),
                                    sample_dist=self.trainer_config.get('sample_distribution', 1),
                                    task_name=self.env_config.get('task', None))
            else:
            
                self.agent = BasePPO(**agent_config)

        
        self.agent.reset(params_key,self.random_key)
        self.step_count=0
        self.episode_lengths=[]
        self.average_reward_per_episode=[]
        self.average_return_per_episode=[]
        self.losses=[]
        self.critic_losses=[]
        self.actor_losses=[]
        self.entropy_losses=[]
        self.sps=[]
        self.result_data=[]
        self.reward_sum=0
        self.statistic_data=dict()
        self.B=self.num_envs*self.rollout_len
        self.log_interval=self.global_config.log_interval
        self.next_log_step=self.log_interval
        self.average_return_per_episode=[]
        self.log_steps = 0
        
        self.best_rewards=[]
        self.mse=[]
        self.last_scaled_diff=[]
        self.last_scaled_obs=[]
        self.scaled_diff=[]
        self.scaled_obs=[]
        self.success=[]
        self.max_dist=[]
        self.actions=[]
        
        self.new_log=[]
        self.log=[]
        self.ratio=[]
        self.ret=[]
        self.vals=[]
        self.advantage=[]
        self.grad_l2=[]
        self.params_l2=[]
        self.params=[]
        self.recon_loss=[]
        self.var_loss=[]
        self.kl_loss=[]
        
        
        
        
        if 'eval_interval' in self.global_config:
            self.eval_interval=self.global_config['eval_interval']
            self.next_eval_step=self.eval_interval
        else:
            self.eval_interval=None
        

    def step(self, **kwargs):
        self.random_key=jax.random.split(self.random_key)[0]

        #Measure steps per second
        start_time=time.time()

        (loss,(value_loss,entropy_loss,actor_loss,rewards), \
            (new_log, log, ratio, ret, vals, advantage, params, var_loss, kl_loss, recon_loss, grad_l2, params_l2), infos)=self.agent.step(self.random_key)
        #Extract info data across all actors and steps
        #Get the leaves of the infos tree where the final_info key is present
        # print("infos", infos)
        
        
        leaves=[info for info in infos if '_final_info' in info]
        # info["batch_mse"] = jnp.mean(jnp.array(self.mse), axis=0)
        #     info["last_scaled_diff"] = avg_scl_diff
        #     info["last_scaled_obs"] = avg_scl_obs
        #     info["scaled_diff"] = jnp.mean(jnp.array(self.scaled_diff), axis=0)
        #     info["scaled_obs"] = jnp.mean(jnp.array(self.scaled_obs), axis=0)
        #     print(info["batch_mse"].shape, info["last_scaled_diff"].shape, info["last_scaled_obs"].shape, info["scaled_diff"].shape, info["scaled_obs"].shape)
        #     info["best_rewards"] = self.best_rewards
        #     info["success"] = ((self.best_rewards[0] - self.y_min) / (self.max_y - self.y_min)) > 0.9
        # Increase the step counter
        self.step_count+=(self.B)
        
        #Iterate over the leaves and extract the final_info data
        for leaf in leaves:
             for k in leaf["final_info"]:
                # start = self.log_steps * self.rollout_len
                # end = start + self.rollout_len
                # print("start", start, "end", end)
                #  print(k["final_info"]["s_rewards"])
                # print("rew", jnp.array(k["final_info"]["s_rewards"],dtype=jnp.float32).shape)
                # s_rewards = jnp.array(k["final_info"]["s_rewards"],dtype=jnp.float32)
                # avg_rew = jnp.mean(s_rewards)
                best_rew = jnp.array(k["final_info"]["best_rewards"],dtype=jnp.float32)
                self.best_rewards.append(jnp.mean(best_rew))
                mse = jnp.array(k["final_info"]["batch_mse"],dtype=jnp.float32)
                self.mse.append(jnp.mean(mse))
                lsd = jnp.array(k["final_info"]["last_scaled_diff"],dtype=jnp.float32)
                self.last_scaled_diff.append(jnp.mean(lsd))
                lso = jnp.array(k["final_info"]["last_scaled_obs"],dtype=jnp.float32)
                self.last_scaled_obs.append(jnp.mean(lso))
                sd = jnp.array(k["final_info"]["scaled_diff"],dtype=jnp.float32)
                self.scaled_diff.append(jnp.mean(sd))
                so = jnp.array(k["final_info"]["scaled_obs"],dtype=jnp.float32)
                
                self.scaled_obs.append(jnp.mean(so))
                success = jnp.array(k["final_info"]["success"],dtype=jnp.bool)
                # print("success", success.shape, success[start:end].shape)
                # [start:end]
                self.success.append(success[0])
                max_x = jnp.array(k["final_info"]["max_x"],dtype=jnp.float32)
                # self.max_dist = jnp.concatenate([self.max_dist,max_x]) 
                
                # print("max_x", max_x.shape, jnp.array(k["final_info"]["max_x"],dtype=jnp.float32).shape, start, end)
                self.max_dist.append(max_x)
                # print("actions", jnp.array(k["final_info"]["actions"],dtype=jnp.float32).shape)
                actions = jnp.array(k["final_info"]["actions"],dtype=jnp.float32)
                # print("actions", actions.shape)
                # self.actions = jnp.concatenate([self.actions,actions]) 
                # print("okat", actions.shape)
                # print("sme", best_rew.shape)
                self.actions.append(actions)
                
           
             for env_info in leaf['final_info'][leaf['_final_info']]:  
                 if 'final_info' in env_info:
                     for key,value in env_info['final_info'].items(): #AutoResetWrapper adds everything in info to final_info after reset along with info from first timestep
                        if isinstance(value, (int, float,)):
                            if key not in self.statistic_data:
                                self.statistic_data[key]=[]
                            self.statistic_data[key].append(value)
                 for key,value in env_info.items():
                     if isinstance(value, (int, float,)):
                         if key not in self.statistic_data:
                             self.statistic_data[key]=[]
                         self.statistic_data[key].append(value)
                
                    #  if k not in self.statistic_data:
                    #      self.statistic_data[k]=[]
                    #  self.statistic_data[k].append(env_info[k])
                #  print("rewards", len(env_info['rewards']))
                 ep_rewards=jnp.array(env_info['rewards'],dtype=jnp.float32)
                 _,average_return_per_episode=average_reward_and_return_in_episode(ep_rewards,self.gamma)
                 self.average_return_per_episode.append(average_return_per_episode)
                #  s_rewards=jnp.array(env_info['s_rewards'],dtype=jnp.float32)
                 
                 

        self.log_steps+=1
        # Log the data
        end_time=time.time()
        self.sps.append(self.B/(end_time-start_time))
        self.losses.append(loss)
        self.critic_losses.append(value_loss)
        self.actor_losses.append(actor_loss)
        self.entropy_losses.append(entropy_loss)
        
        self.new_log.append(new_log)
        self.log.append(log)
        self.ratio.append(ratio)
        self.ret.append(ret)
        self.vals.append(vals)
        self.advantage.append(advantage)
        self.grad_l2.append(grad_l2)
        self.params_l2.append(params_l2)
        self.params.append(params)
        self.var_loss.append(var_loss)
        self.kl_loss.append(kl_loss)
        self.recon_loss.append(recon_loss)
        
        self.reward_sum+=rewards.sum()
        if self.step_count>=self.next_log_step:
            #Calculate the mean of the elements in statistic_data and log them, finally clear the statistic_data
            metrics={}
            for key in self.statistic_data.keys():
                agg_value=np.mean(self.statistic_data[key])
                metrics={**metrics,key:agg_value}
                self.statistic_data[key]=[]
            self.next_log_step+=self.log_interval
            critic_loss=np.mean(self.critic_losses)
            actor_loss=np.mean(self.actor_losses)
            entropy_loss=np.mean(self.entropy_losses)
            loss=np.mean(self.losses)
            reward_mean=float(self.reward_sum/self.log_interval)
            return_mean=np.mean(self.average_return_per_episode)
            
            n_log_p = np.mean(self.new_log)
            log_p = np.mean(self.log)
            m_ratio = np.mean(self.ratio)
            m_ret = np.mean(self.ret)
            m_vals = np.mean(self.vals)
            m_advantage = np.mean(self.advantage)
            m_grad_l2 = np.mean(self.grad_l2)
            m_params_l2 = np.mean(self.params_l2)
            # m_pol_par = jnp.mean(jnp.array(self.params), axis=0)
            m_var_l = np.mean(self.var_loss)
            m_kl_l = np.mean(self.kl_loss)
            m_recon_l = np.mean(self.recon_loss)
            
        
            
            
            
            
            best_mean=np.mean(self.best_rewards)
            scaled_diff_mean=np.mean(self.scaled_diff)
            last_scaled_diff_mean=np.mean(self.last_scaled_diff)
            scaled_obs_mean=np.mean(self.scaled_obs)
            last_scaled_obs_mean=np.mean(self.last_scaled_obs)
            mse_mean=np.mean(self.mse)
            success_mean=np.mean(self.success)
            # print("actos", jnp.array(self.actions).shape)
            # print("max_dist", jnp.array(self.actions).flatten().shape)
            # print("max_dist", jnp.array(self.max_dist).flatten().shape, jnp.array(self.actions).shape)
            
            self.actions = jnp.array(self.actions)
            actions_dim = self.actions.reshape(-1, self.actions.shape[-1])
            actions_dims = {}
            for i in range(actions_dim.shape[-1]):
                actions_dims[f"action/actions_{i}"] = wandb.Histogram(actions_dim[:, i])
            
            # split = m_pol_par.shape[0] // 2
            
            # for i in range(split):
            #     actions_dims[f"action/mu_{i}"] = wandb.Histogram(m_pol_par[i])
            #     actions_dims[f"action/std_{i}"] = wandb.Histogram(m_pol_par[split + 1])
                
            # act_dist = wandb.Histogram(jnp.array(self.actions).flatten())
            max_dist = wandb.Histogram(jnp.array(self.max_dist).flatten())
            
            mean_sps=np.mean(self.sps)
            self.reward_sum=0
            self.critic_losses=[]
            self.actor_losses=[]
            self.entropy_losses=[]
            self.losses=[]
            self.sps=[]
            
            self.best_rewards=[]
            self.mse=[]
            self.last_scaled_diff=[]
            self.last_scaled_obs=[]
            self.scaled_diff=[]
            self.scaled_obs=[]
            self.success=[]
            self.max_dist=[]
            self.actions=[]
            
            self.new_log=[]
            self.log=[]
            self.ratio=[]
            self.ret=[]
            self.vals=[]
            self.advantage=[]
            self.grad_l2=[]
            self.params_l2=[]
            
          
           
            # for k in scatter.keys():
            #     scatter[k] = scatter[k].tolist()
        
            
            
            # table = wandb.Table(data=[[scatter["actions"][i], scatter["rewards"][i]] for i in range(len(scatter["actions"]))], columns=["actions", "rewards"])
            # a = wandb.plot.scatter(table, "actions", "rewards", title="Rewards over actions")
            # table = wandb.Table(data=[[scatter["actions"][i], scatter["critic_preds"][i]] for i in range(len(scatter["actions"]))], columns=["actions", "critic_pred"])
            # b = wandb.plot.scatter(table, "actions", "critic_pred", title="Critics over actions")
            # table = wandb.Table(data=[[scatter["actions"][i], scatter["advantages"][i]] for i in range(len(scatter["actions"]))], columns=["actions", "advantages"])
            # c = wandb.plot.scatter(table, "actions", "advantages", title="Advatages over actions")
            # table = wandb.Table(data=[[scatter["critic_preds"][i], scatter["rewards"][i]] for i in range(len(scatter["critic_preds"]))], columns=["critic_preds", "rewards"])
            # d = wandb.plot.scatter(table, "critic_preds", "rewards", title="critics over rewards")
            # table = wandb.Table(data=[[scatter["actions"][i], scatter["glambdas"][i]] for i in range(len(scatter["actions"]))], columns=["actions", "glambdas"])
            # e = wandb.plot.scatter(table, "actions", "glambdas", title="glambdas over actions")
            
            
            
           

            # print("scatter", scatter["actions"].shape, scatter["rewards"].shape, scatter["critic_preds"].shape, scatter["advantages"].shape, len(scatter["actions"]))
           
            
            self.average_return_per_episode=[]
            metrics={'step':self.step_count,'sps':mean_sps,'loss/loss':loss,'loss/critic_loss':critic_loss,
                                    'loss/actor_loss':actor_loss,'loss/entropy_loss':entropy_loss,'env_metrics/mean_reward':reward_mean,
                                    'env_metrics/return_per_episode':return_mean, 
                                    
                                    'env_metrics/best action':best_mean, 
                                    'env_metrics/scaled_diff':scaled_diff_mean, 'env_metrics/last_scaled_diff':last_scaled_diff_mean, 'env_metrics/scaled_obs':scaled_obs_mean, 
                                    'env_metrics/last_scaled_obs':last_scaled_obs_mean, 'env_metrics/mse':mse_mean, 'env_metrics/success':success_mean, #'env_metrics/actions':act_dist, 
                                    'env_metrics/max_dist':max_dist,
                                    'loss/new log prob':n_log_p, 'loss/log prob':log_p, 'loss/ratio':m_ratio, 'loss/return':m_ret, 'loss/value predicition':m_vals, 'loss/advantage':m_advantage, 
                                    'loss/grad l2':m_grad_l2, 'loss/params l2':m_params_l2, "loss/variational_loss":m_var_l, "loss/kl_loss":m_kl_l, "loss/recon_loss":m_recon_l, 
                                   # 'advantage/reward':a,'advantage/critic_preds':b,'advantage/advantages':c, 'advantage/critic_rewards':d, 'advantage/glambdas':e,
                                   
                                    
                                    
                                    **metrics
                                    }
            metrics = metrics | actions_dims
            self.result_data.append(metrics)
        else:
            metrics=None
        if self.eval_interval is not None and self.step_count>=self.next_eval_step:
            self.next_eval_step+=self.eval_interval
            avg_episode_len,avg_episode_return,rollouts, eval_stats=self.agent.evaluate(self.random_key,self.global_config['eval_episodes'])
            # print("rollouts", rollouts.shape, avg_episode_len, avg_episode_return)
            # rollouts=np.concatenate(rollouts,axis=0)
            # rollouts = np.ones((5))
            # column_names = [f"dim_{i+1}" for i in range(rollouts.shape[1] - 2)] + ["scaled_diff", "reward"]
            # df = pd.DataFrame(jnp.array(rollouts), columns=column_names)
            # print(df)
            if metrics is None:
                metrics={}
            metrics['step']=self.step_count
            metrics['eval/eval_avg_episode_len']=float(avg_episode_len)
            metrics['eval/eval_avg_episode_return']=float(avg_episode_return)
            metrics['eval/avg_reward']=float(eval_stats['avg_rew'])
            metrics['eval/regret']=float(eval_stats['regret'])
            metrics['eval/best_action']=float(eval_stats['best'])
            metrics['eval/succes rate']=float(eval_stats['success'])
            # metrics['rollouts']=wandb.Video(rollouts, fps=self.global_config.get('record_fps',5), format="gif")
            # metrics['rollouts']=wandb.Table(dataframe=df)
        return loss,metrics,self.step_count
    

    def get_summary_table(self):
        return pd.DataFrame(self.result_data).to_json(default_handler=str)
    
