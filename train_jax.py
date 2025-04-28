from cmath import isnan
from re import U
import sys
sys.path.append('./')
# sys.path.append('../') # Adjust if necessary based on your project structure

import numpy as np
import random
# import json # Not used directly here, but maybe by hydra/wandb
import argparse # Not used directly here, but maybe by hydra
import wandb
import jax
import hydra
# import jax.numpy as jnp # Not used directly here
# import itertools # Not used directly here

# --- Import the NEW JAX-based trainer ---
# Assuming it's saved in src/trainers/trainers_control_jax_refactored.py
from src.trainers.trainers_control_jax import ControlTrainerJaxRefactored
# Remove or keep ControlTrainer if needed for other tasks
# from src.trainers.trainers_control import ControlTrainer

from omegaconf import DictConfig, OmegaConf
# import pandas as pd # Used only in get_summary_table in trainer
from tqdm import tqdm
# import multiprocessing as mp # Not used directly here
# from concurrent.futures import ProcessPoolExecutor # Not used directly here
# from multiprocessing import Pool # Not used directly here
# import multiprocessing as mp # Not used directly here
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
# import ast # Not used directly here

logger = logging.getLogger(__name__)
# Configure logging if needed, e.g., logging.basicConfig(level=logging.INFO)

# --- Updated Task Mapping ---
task_to_trainer={
    # 'minigrid_pixel': ControlTrainer, # Keep old ones if still used
    # 'minigrid_onehot': ControlTrainer,
    # ... other old tasks ...
    'jax': ControlTrainerJaxRefactored, # Map 'jax' task to the new trainer
}

@hydra.main(version_base=None, config_path="config", config_name="sweep_test")
def main(config: DictConfig):

    # --- Configuration & Setup ---
    # logger.info("Starting Job for Config:\n"+str(OmegaConf.to_yaml(config)))
    # Construct run name, project name, tags based on config
    norm = "-norm:" + config.trainer.dist_model if config.trainer.dist_model != "standard" else ""
    rand = "rand_" if config.task.random == True else "" # Assuming 'random' key exists
    # Ensure batches is usable in name (convert list to string if needed)
    batch_str = str(config.task.batches)
    run_name = f"{rand}method {config.task.task}-batch {batch_str}-samples {config.task.total_episode_samples}{norm}-seed {config.seed}- {config.run_name}"

    env_str = "train" + str(config.task.env_train) + "-test" + str(config.task.env_test)
    project_name = f"env {env_str}-dim {config.task.action_dim}-bounds {config.task.bounds}-{config.project_name}"
    tags = config.tags.split(',') if config.tags is not None else []
    # Example tag: tags = [f"m{config.task.task}b{batch_str}s{config.task.total_episode_samples}"]

    # --- Initialize WandB ---
    run = None
    if config.use_wandb:
        # Choose start_method based on platform or leave default
        settings = wandb.Settings(start_method="thread") # Use "thread" for better cross-platform compatibility?
        try:
            run = wandb.init(project=project_name, name=run_name, tags=tags, config=OmegaConf.to_container(config, resolve=True), settings=settings)
            logger.info(f"WandB run initialized: {run.name} (ID: {run.id})")
        except Exception as e:
             logger.error(f"WandB initialization failed: {e}")
             run = None


    # --- Basic Setup ---
    key = jax.random.key(config.seed)
    # Set seeds for other libraries if used elsewhere, JAX handles its own seeding via keys
    np.random.seed(config.seed)
    random.seed(config.seed)

    trainer_config = config.trainer
    env_config = config.task # This should be the DictConfig object for env parameters

    # --- Instantiate Trainer ---
    kwargs={'trainer_config': trainer_config,
            'env_config': env_config, # Pass the DictConfig directly
            'global_args': config, # Pass the top-level config
            # 'seed': config.seed,   # Seed is mainly for JAX key now
            'key': key,            # Pass the initial JAX key
            'wandb_run': run}

    # Select the trainer class based on the task config
    if env_config['task'] not in task_to_trainer:
        raise ValueError(f"Unsupported task '{env_config['task']}'. Available: {list(task_to_trainer.keys())}")

    trainer_class = task_to_trainer[env_config['task']]
    logger.info(f"Using trainer: {trainer_class.__name__}")

    # --- Check if using the refactored trainer ---
    if trainer_class == ControlTrainerJaxRefactored:
        # Initialize the refactored trainer
        
        # print("all the kwar", kwargs) # Debugging line to check kwargs
        
        trainer = ControlTrainerJaxRefactored(**kwargs)
        logger.info("Initialized ControlTrainerJaxRefactored.")

        # --- Run Training (Loop is now inside trainer.train()) ---
        logger.info("Starting training loop...")
        with logging_redirect_tqdm(): # Redirect tqdm progress bar logging
            trainer.train() # Call the internal training loop
        logger.info("Training finished.")

        # --- Optional: Log final summary table ---
        if config.use_wandb and run:
            try:
                summary_json_str = trainer.get_summary_table()
                # Potentially log as artifact or directly if small enough
                # wandb.log({"final_summary_table": summary_json_str})
                logger.info("Final summary table generated (not logged to wandb in this example).")
            except Exception as e:
                logger.error(f"Failed to get or log summary table: {e}")

    else:
        # --- Original Training Loop (for non-refactored trainers) ---
        logger.info("Using original ControlTrainer (Gymnasium-based).")
        trainer = trainer_class(**kwargs)
        pbar = tqdm(total=config.steps, desc="Training Steps")
        step_count = 0
        last_step_count = 0
        with logging_redirect_tqdm():
            while True:
                # Assuming original trainer.step() returns loss, metrics, step_count
                loss, metrics, current_step_count = trainer.step()
                step_delta = current_step_count - last_step_count
                if step_delta > 0:
                    pbar.update(n=step_delta)
                last_step_count = current_step_count

                if metrics is not None:
                    log_step = metrics.get('step', current_step_count) # Use step from metrics if available
                    # logger.info(f"Seed: {config.seed} Steps: {log_step} Metrics: {metrics}") # Log less verbosely
                    if run:
                        run.log({'seed': config.seed, **metrics}, step=log_step)

                if current_step_count >= config.steps:
                    break
        pbar.close()
        logger.info("Training finished.")
        # Log summary table if original trainer provides it
        if config.use_wandb and run and hasattr(trainer, 'get_summary_table'):
             summary_json_str = trainer.get_summary_table()
             # wandb.log({"final_summary_table": summary_json_str})


    # --- Finish WandB Run ---
    if run:
        wandb.finish()
        logger.info("WandB run finished.")

if __name__=='__main__':
    # Start method selection might not be needed if not using multiprocessing explicitly here
    # Consider defaults or configure based on platform if issues arise with libraries.
    main()