import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from customenv import CustomEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
import torch as th
import torch.nn as nn
from sb3_contrib import MaskablePPO
import torch as th
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from multiprocessing import freeze_support
import os
from sb3_contrib.common.maskable.evaluation import evaluate_policy
import json
import time
from datetime import datetime

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def make_env(rank):
    def _init():
        env = CustomEnv(25, "Random",normalize_rewards=True,p=3)
        env = Monitor(env, filename=f"25_Random_rank{rank}_monitor.csv", allow_early_resets=True, override_existing=False)
        return env
    return _init


def log_training_metadata(
    model, 
    env, 
    env_params=None,  # New parameter for environment settings
    starting_model=None, 
    new_model_name=None, 
    timesteps=0,
    training_time=0,
    initial_performance=None,
    final_performance=None,
    best_performance=None
):
    """Create a metadata file with training details."""
    
    # Create metadata directory if it doesn't exist
    metadata_dir = "./training_metadata"
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use provided env_params or default
    if env_params is None:
        env_params = {
            "env_type": "CustomEnv",
            "graph_size": 25,
            "instance_type": "Random",
            "normalize_rewards": True,
            "p": 3
        }
    
    # Extract model hyperparameters
    if hasattr(model, "learning_rate"):
        if callable(model.learning_rate):
            lr = "schedule"
        else:
            lr = float(model.learning_rate)
    else:
        lr = "unknown"
        
    hyperparams = {
        "learning_rate": lr,
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "ent_coef": float(model.ent_coef),
        "vf_coef": float(model.vf_coef),
        "policy_type": model.policy.__class__.__name__,
        "num_envs": env.num_envs
    }
    
    # Create metadata dictionary
    metadata = {
        "timestamp": timestamp,
        "starting_model": starting_model or "None (new model)",
        "new_model_name": new_model_name,
        "total_timesteps": timesteps,
        "training_duration_seconds": training_time,
        "environment_parameters": env_params,
        "model_hyperparameters": hyperparams,
        "hardware": {
            "device": str(model.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        },
        "performance": {
            "initial": initial_performance,
            "final": final_performance,
            "best": best_performance
        }
    }
    
    # Save metadata to file
    filename = f"{metadata_dir}/training_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Training metadata saved to {filename}")
    return filename


def main(
    # Environment parameters
    graph_size=25,
    instance_type="Clustered",
    normalize_rewards=True,
    env_class=CustomEnv,
    env_kwargs=None,
    
    # Training parameters
    num_envs=None,
    model_name_prefix="masked_ppo",
    timesteps=5000000,
    learning_rate=1e-4,
    n_steps=8192,
    batch_size=256,
    n_epochs=10,
    gamma=0.999,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    
    # Evaluation parameters
    eval_freq=10000,
    n_eval_episodes=600,
    n_final_eval_episodes=3000,
    
    # Loading existing model
    existing_model_path=None,
    fine_tuning_lr=5e-4
):
    """
    Main training function with configurable parameters.
    
    Args:
        # Standard environment parameters
        graph_size: Size of the graph for environment
        instance_type: Type of instances ("Random", "Euclidean", etc.)
        normalize_rewards: Whether to normalize rewards in the environment
        p: Parameter for precedence constraints
        env_class: The environment class to use (defaults to CustomEnv)
        env_kwargs: Additional keyword arguments to pass to the environment
        
        # (... rest of docstring remains the same)
    """
    
    # Set number of environments if not specified
    if num_envs is None:
        num_envs = os.cpu_count()
        
    print(f"Using {num_envs} parallel environments")
    
    # Initialize env_kwargs if not provided
    if env_kwargs is None:
        env_kwargs = {}
    
    # Set up logging FIRST - before creating environments
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Define make_env function here, after log_dir is defined
    def make_env(rank):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        def _init():
            # Create environment with all parameters
            combined_kwargs = {
                "N_GRAPH_SIZE": graph_size,
                "type": instance_type,
                "normalize_rewards": normalize_rewards,
                **env_kwargs  # Additional environment-specific parameters
            }
            
            env = env_class(**combined_kwargs)
            
            # Use the log_dir from outer scope
            filename = f"{log_dir}/{graph_size}_{instance_type}_rank{rank}_{timestamp}_monitor.csv"
            
            env = Monitor(env, 
                         filename=filename,
                         allow_early_resets=True, 
                         override_existing=False)
            return env
        return _init
    
    # Now create environments after log_dir and make_env are defined
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    eval_env = SubprocVecEnv([make_env(i + num_envs) for i in range(4)])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up logging
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create output model name with parameters
    model_id = f"{graph_size}_{instance_type.lower()}_{model_name_prefix}"

    # Set up evaluation callback
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/results",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=n_eval_episodes,
        verbose=1
    )
    
    initial_performance = None

    try:
        if existing_model_path:
            # Try to load existing model
            print(f"Loading existing model from {existing_model_path}")
            model = MaskablePPO.load(
                existing_model_path,
                env=env,
                device=device,
                verbose=0
            )
            print("Successfully loaded model")

            print("Evaluating loaded model...")
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_final_eval_episodes, deterministic=True)
            initial_performance = {"mean_reward": float(mean_reward), "std_reward": float(std_reward)}
            print(f"Initial Performance: {mean_reward} +- {std_reward}")
            
            # Adjust learning rate for fine-tuning if specified
            if fine_tuning_lr:
                model.learning_rate = fine_tuning_lr
                print(f"Adjusted learning rate to {fine_tuning_lr} for fine-tuning")
        else:
            raise FileNotFoundError("No existing model specified, creating new one")
            
    except (FileNotFoundError, Exception) as e:
        print(f"Could not load model: {e}")
        print("Creating new model instead")
        
        # Create a new model if loading fails
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            device=device,
            verbose=0
        )
    
    # Continue training
    print("Starting training...")
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    training_time = time.time() - start_time
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to new path with timestamp
    new_model_path = f"{model_id}_{timestamp}"
    print(f"Saving model to {new_model_path}")
    model.save(new_model_path)
    
    print("Training complete!")

    print("Evaluating final model...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_final_eval_episodes, deterministic=True)
    final_performance = {"mean_reward": float(mean_reward), "std_reward": float(std_reward)}
    print(f"Final Performance: {mean_reward} +- {std_reward}")

    best_performance = None
    try:
        best_model = MaskablePPO.load(f"{log_dir}/best_model/best_model", env=eval_env)
        print("Evaluating best model from training...")
        best_mean_reward, best_std_reward = evaluate_policy(best_model, eval_env, n_eval_episodes=n_final_eval_episodes)
        best_performance = {"mean_reward": float(best_mean_reward), "std_reward": float(best_std_reward)}
        print(f"Best model performance: {best_mean_reward:.2f} ± {best_std_reward:.2f}")
    except Exception as e:
        print(f"Could not evaluate best model: {e}")
    
    # Create environment parameters dictionary for logging
    env_params = {
        "env_type": env_class.__name__,
        "graph_size": graph_size,
        "instance_type": instance_type,
        "normalize_rewards": normalize_rewards
    }
    # Add any additional env_kwargs to env_params
    env_params.update(env_kwargs)
    
    # Log training metadata
    log_training_metadata(
        model=model,
        env=env,
        env_params=env_params,  # Pass environment parameters explicitly
        starting_model=existing_model_path,
        new_model_name=new_model_path,
        timesteps=timesteps,
        training_time=training_time,
        initial_performance=initial_performance,
        final_performance=final_performance,
        best_performance=best_performance
    )
    
    return model, new_model_path, final_performance

if __name__ == '__main__':
    freeze_support()
    
    # Example usage with default parameters
    main(
    # Environment parameters
    graph_size=25,
    instance_type="ClusteredWithRandomAsymmetry",
    normalize_rewards=True,
    env_kwargs = {'p': 2},
    
    # Training parameters
    num_envs=16,
    model_name_prefix="masked_ppo",
    timesteps=20000000,
    learning_rate=5e-4,
    n_steps=8192,
    batch_size=256,
    n_epochs=10,
    gamma=1,
    gae_lambda=0.95,
    ent_coef=0.02,
    vf_coef=0.5,
    
    # Evaluation parameters
    eval_freq=10000,
    n_eval_episodes=600,
    n_final_eval_episodes=10000,
    
    # Loading existing model
    existing_model_path="25_clusteredwithrandomasymmetry_masked_ppo_20250805_011148.zip", #"25_euclidic_masked_ppo_20250716_201002.zip",  # Set to None to create a new model
    fine_tuning_lr=5e-4
)