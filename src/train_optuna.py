import optuna
from optuna.integration import TensorBoardCallback
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from customenv import CustomEnv
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Simplified Feature Extractor
class OptimizedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, hidden_sizes=[512, 256]):
        super().__init__(observation_space, features_dim)
        
        input_dim = np.prod(observation_space.shape)
        layers = []
        
        # Input layer
        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], features_dim))
        layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, observations):
        return self.network(observations)

def mask_fn(env):
    if hasattr(env,"env"):
        return env.env.get_action_mask()
    else:
        return env.get_action_mask()

def create_optimized_environment(size=15, difficulty="Random"):
    """Create environment for optimization."""
    base_env = CustomEnv(size, difficulty, p=0.05)
    return ActionMasker(Monitor(base_env), mask_fn)

def objective(trial):
    """Optuna objective function."""
    
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_epochs = trial.suggest_int("n_epochs", 5, 15)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.25, 1.0)
    
    # Feature extractor parameters
    features_dim = trial.suggest_categorical("features_dim", [128, 256, 512])
    hidden_size_1 = trial.suggest_categorical("hidden_size_1", [256, 512, 1024])
    hidden_size_2 = trial.suggest_categorical("hidden_size_2", [128, 256, 512])
    
    try:
        # Create environment
        env = create_optimized_environment()
        
        # Create model
        model = MaskablePPO(
            "MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=0,
            policy_kwargs=dict(
                features_extractor_class=OptimizedFeatureExtractor,
                features_extractor_kwargs=dict(
                    features_dim=features_dim,
                    hidden_sizes=[hidden_size_1, hidden_size_2]
                )
            )
        )
        
        # Quick training
        model.learn(total_timesteps=100000)
        
        # Evaluate
        eval_env = create_optimized_environment()
        mean_reward, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        
        # Cleanup
        env.close()
        eval_env.close()
        del model
        
        return mean_reward
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return -1000  # Return poor score for failed trials

def run_optimization():
    """Run the hyperparameter optimization."""
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler()
    )
    
    # Optimize
    study.optimize(objective, n_trials=30, timeout=7200)  # 2 hours
    
    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")
    
    return study.best_params

def train_with_best_params(best_params):
    """Train final model with optimized parameters."""
    
    # Create full-size environment
    env = ActionMasker(
        Monitor(CustomEnv(25, "Clustered", **{"p_intra": 0.1, "p_inter": 0.02})),
        mask_fn
    )
    
    # Create model with best parameters
    model = MaskablePPO(
        "MlpPolicy",
        env=env,
        learning_rate=best_params["learning_rate"],
        n_steps=best_params["n_steps"],
        batch_size=best_params["batch_size"],
        n_epochs=best_params["n_epochs"],
        gamma=best_params["gamma"],
        gae_lambda=best_params["gae_lambda"],
        clip_range=best_params["clip_range"],
        ent_coef=best_params["ent_coef"],
        vf_coef=best_params["vf_coef"],
        verbose=1,
        policy_kwargs=dict(
            features_extractor_class=OptimizedFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=best_params["features_dim"],
                hidden_sizes=[best_params["hidden_size_1"], best_params["hidden_size_2"]]
            )
        )
    )
    
    # Train final model 
    model.learn(total_timesteps=2000000)
    model.save("optimized_tsp_model")
    
    return model

# Run the complete optimization process
if __name__ == "__main__":
    print("Starting hyperparameter optimization...")
    best_params = run_optimization()
    
    print("\nTraining final model with optimized parameters...")
    final_model = train_with_best_params(best_params)
    
    print("Optimization complete!")