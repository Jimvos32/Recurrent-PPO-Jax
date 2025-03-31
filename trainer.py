from cmath import isnan
from re import U
import sys
sys.path.append('./')
sys.path.append('../')



import numpy as np
import random
import json
import argparse
import wandb
import jax
import hydra
import jax.numpy as jnp
import itertools
from src.trainers.trainers_control import ControlTrainer
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool 
import multiprocessing as mp
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
import ast

 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

task_to_trainer={
    'minigrid_pixel':ControlTrainer,
    'minigrid_onehot':ControlTrainer,
    'sampling':ControlTrainer,
    'multi':ControlTrainer,
    'batch':ControlTrainer,
    'multibatch':ControlTrainer,
    'expanded_samp':ControlTrainer,
    'multidim':ControlTrainer,
    'masked':ControlTrainer,
    'gen_gmm': ControlTrainer,
    'cor_gmm': ControlTrainer,
    'full_params': ControlTrainer,
    'vae': ControlTrainer,
}

@hydra.main(version_base=None, config_path="config", config_name="sweep_test")
def main(config: DictConfig):
    # print("beginninh:", config.task.batches)

    # if isinstance(config.task.batches, list):
    #     print("Batches are already a list")
    #     config.task.batches = [ast.literal_eval(x) if isinstance(x, str) else x for x in config.task.batches]
    
    # print("Batches used for this run:", config.task.batches)
    print("Resolved task.batches =", config.task.batches)
    print("Resolved task.max_episode_steps =", config.task.max_episode_steps)

    print(config.task)
    # logger.info("Starting Job for Config:\n"+str(OmegaConf.to_yaml(config)))
    norm = "-norm:" + config.trainer.dist_model if config.trainer.dist_model != "standard" else ""
    run_name = "method " + config.task.task +"-batch "+  str(config.task.batches) + "-samples "+ str(config.task.total_episode_samples) + norm +"-seed "+str(config.seed) + "- " + config.run_name
    degree = config.task.num_oscillations if config.task.env == "cosine" else config.task.degree

    project_name = "env " + config.task.env + "-dim " + str(config.task.action_dim) + "-deg " + str(degree) + "-bounds " + str(config.task.bounds) + "-" + config.project_name
    tags=config.tags.split(',') if config.tags is not None else []
    # project_name = "debugging"
    
    tags = [("m" + config.task.task + "b" + str(config.task.batches) + "s" + str(config.task.total_episode_samples))]
    print("Tags:", tags)
    
    if config.use_wandb and sys.platform=='win32':
        run = wandb.init(project=project_name,name=run_name,tags=tags,settings=wandb.Settings(start_method="spawn"),config=OmegaConf.to_container(config))
    elif config.use_wandb and sys.platform!='win32':
        run = wandb.init(project=project_name,name=run_name,tags=tags,settings=wandb.Settings(start_method="fork"),config=OmegaConf.to_container(config))
    else:
        run=None
    key=jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    trainer_config=config.trainer
    env_config=config.task
    
    
    env_config['batches'] = config.task.batches
    env_config['max_episode_steps'] = env_config['total_episode_samples'] // env_config['batches'][0]
    print("env", env_config)
   
    #Train the model
    kwargs={'global_args':config,'trainer_config':trainer_config,'env_config':env_config,
            'seed':config.seed,'key':key,'wandb_run':run}
    # print(kwargs['the_test'])
   
    
    trainer=task_to_trainer[env_config['task']](**kwargs)
    pbar = tqdm(total=config.steps)
    step_count=0
    last_step_count=0
    with logging_redirect_tqdm():
        while True:
            loss,metrics,step_count=trainer.step()
            pbar.update(n=step_count-last_step_count)
            last_step_count=step_count
            if metrics is not None:
                logger.info("Seed: "+str(config.seed)+" Steps: "+str(step_count)+" Metrics: "+str(metrics))
                if config.use_wandb: run.log({'seed':config.seed,**metrics
                                                },step=step_count)


            if step_count>=config.steps:
                break
    pbar.close()
    if config.use_wandb:
        wandb.finish()
        #Need to do something about logging val_metric


if __name__=='__main__':
    # if sys.platform=='win32':
    #     mp.set_start_method('spawn')
    # else:
    #     mp.set_start_method('fork')
    main()

