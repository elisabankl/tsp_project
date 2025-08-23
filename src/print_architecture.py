# Add this to your code after loading/creating the model
from sb3_contrib import MaskablePPO


def print_network_architecture(model):
    """Print the architecture of a PPO network."""
    print("\n===== POLICY NETWORK ARCHITECTURE =====")
    
    # Print policy details
    print(f"\nPolicy type: {type(model.policy).__name__}")
    
    # Print actor network (action distribution)
    print("\n--- Actor Network ---")
    print(model.policy.action_net)
    total_params_actor = sum(p.numel() for p in model.policy.action_net.parameters())
    print(f"Total parameters (Actor): {total_params_actor:,}")
    
    # Print value network (critic)
    print("\n--- Critic Network ---")
    print(model.policy.value_net)
    total_params_critic = sum(p.numel() for p in model.policy.value_net.parameters())
    print(f"Total parameters (Critic): {total_params_critic:,}")
    
    # Print feature extractor if present
    if hasattr(model.policy, "features_extractor"):
        print("\n--- Features Extractor ---")
        print(model.policy.features_extractor)
        total_params_features = sum(p.numel() for p in model.policy.features_extractor.parameters())
        print(f"Total parameters (Features): {total_params_features:,}")
    
    # Print shared layers if present
    if hasattr(model.policy, "mlp_extractor"):
        print("\n--- MLP Extractor (Shared Layers) ---")
        print(model.policy.mlp_extractor)
        
        if hasattr(model.policy.mlp_extractor, "policy_net"):
            print("\nPolicy Branch:")
            print(model.policy.mlp_extractor.policy_net)
        
        if hasattr(model.policy.mlp_extractor, "value_net"):
            print("\nValue Branch:")
            print(model.policy.mlp_extractor.value_net)
            
        total_params_mlp = sum(p.numel() for p in model.policy.mlp_extractor.parameters())
        print(f"Total parameters (MLP Extractor): {total_params_mlp:,}")
    
    # Print total parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\nTotal network parameters: {total_params:,}")

model_path = "25_random_masked_ppo_20250625_044937.zip"  # Replace with your model path
model = MaskablePPO.load(model_path, verbose=0)
print_network_architecture(model)
print(model.policy)