import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from matrices import generate_distance_matrix, generate_precedence_matrix, generate_cost_matrix
from customenv import CustomEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
#from torch_geometric.nn import GCNConv
#from torch_geometric.nn import GCNConv
#from torch_geometric.data import Data
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from multiprocessing import freeze_support


device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Define GNN layers
        self.conv1 = GCNConv(1, 64)  # Input features per node = 1 (e.g., node degree or dummy feature)
        self.conv2 = GCNConv(64, features_dim//25)

    def forward(self, observations):
        """
        Convert the edge length matrix into a graph representation and apply GNN layers.
        :param observations: Edge length matrix (batch_size, num_nodes, num_nodes)
        """
        batch_size, _, num_nodes,_ = observations.shape
        print("observations shape: ",observations.shape)
        print(batch_size)
        print("num nodes:  ",num_nodes)

        # Create node features (dummy features, e.g., all ones)
        node_features = th.ones((batch_size * num_nodes, 1), device=observations.device)
        print("Node features: ",node_features.shape)

        # Create edge indices for a fully connected graph
        edge_index = th.combinations(th.arange(num_nodes, device=observations.device), r=2).T
        edge_index = th.cat([edge_index, edge_index.flip(0)], dim=1)  # Add reverse edges for undirected graph

        # Flatten the edge length matrix into edge attributes
        edge_lengths = observations[:,0,:,:].reshape(batch_size, -1)  # Flatten each matrix in the batch
        print("Edge lenghts: ",edge_lengths.shape)
        # Process each graph in the batch
        outputs = []
        for i in range(batch_size):
            # Create a PyTorch Geometric Data object for each graph
            data = Data(x=node_features[i * num_nodes:(i + 1) * num_nodes],
                        edge_index=edge_index,
                        edge_attr=edge_lengths[i])

            # Apply GNN layers
            print("x Shape:   ",data.x.shape)

            x = th.relu(self.conv1(data.x, data.edge_index))
            x = self.conv2(x, data.edge_index)
            outputs.append(x)

        
        # Stack outputs for the batch
        node_embeddings = th.stack(outputs, dim=0)  # Shape: [batch_size, num_nodes, features_dim]
        print("Node embeddings shape: ", node_embeddings.shape)

        # Flatten the node embeddings to [batch_size, num_nodes * features_dim]+
        graph_embeddings = node_embeddings.view(batch_size, -1)
        print("Graph embeddings shape: ", graph_embeddings.shape)

        return graph_embeddings

# Function to apply action masks
def mask_fn(env):
    if hasattr(env,"env"):
        return env.env.action_masks()
    else:
        return env.action_masks()

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

class GNNPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(GNNPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=GNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=10*25)
        )

def make_env():
    def _init():
        env = CustomEnv(25, "Euclidic",normalize_rewards=True,p=0.3)
        env = Monitor(env, filename=f"25_Euclidic_monitor.csv", allow_early_resets=True, override_existing=False)
        env = ActionMasker(env, mask_fn)
        return env
    return _init

def main():
    num_envs = 8
    env = DummyVecEnv([make_env()])

    model = MaskablePPO(
    policy = "MlpPolicy",
    env = env,
    learning_rate=1e-4,
    n_steps=8192,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    device=device,
    verbose=1
    )

    model.learn(total_timesteps=2000000)
    model.save("25_euclidic_masked_ppo")

if __name__ == '__main__':
    #freeze_support()
    main()

    
# Create the training environment and wrap it with Monitor and ActionMasker
#env = ActionMasker(Monitor(CustomEnv(25,"Clustered",normalize_rewards = True), filename="25_clustered_monitor.csv", allow_early_resets=True,override_existing=False), mask_fn)
#env = DummyVecEnv([lambda: ActionMasker(Monitor(CustomEnv(25), filename="masked_ppo_25_non_normalized_monitor.csv", allow_early_resets=True), mask_fn)])
#env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Create the evaluation environment and wrap it with Monitor and ActionMasker
#eval_env = ActionMasker(Monitor(CustomEnv(25,"Clustered",normalize_rewards = True), filename="25_clustered_eval_monitor.csv", allow_early_resets=True,override_existing=False), mask_fn)

#eval_env = DummyVecEnv([lambda: ActionMasker(Monitor(CustomEnv(25), filename="masked_ppo_eval_non_normalized_monitor.csv", allow_early_resets=True), mask_fn)])
#eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

# Set up the MaskableEvalCallback
#eval_callback = MaskableEvalCallback(eval_env, best_model_save_path='./logs_non_normalized/',
 #                                    log_path='./logs_non_normalized/', eval_freq=50000,
 #                                    n_eval_episodes=200, deterministic=True, render=False)

# Load the existing model if it exists, otherwise create a new one
"""try:
    model = MaskablePPO.load("25_clustered", env=env, verbose=0)
    print("Loaded existing model.")
except FileNotFoundError:
    model = MaskablePPO(
    "MlpPolicy",
    env=env,
    verbose=1
)
    print("Created new model.")"""
#print(model.policy)
#print(model.policy.features_extractor)
#print(model.policy.action_dist)
# Train the model
#model.learn(total_timesteps=500000, callback=eval_callback)

# Save the model
#model.save("25_clustered")