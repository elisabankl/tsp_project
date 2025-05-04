import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from customenv import CustomEnv

def generate_random_solution(env):
    obs, _ = env.reset(fixed_instance=True)
    done = False
    truncated = False
    episode_rewards = 0
    while not (done or truncated):
        # Get the action masks
        action_masks = env.action_masks()
        # Randomly select an allowed action
        allowed_actions = np.flatnonzero(action_masks)
        action = np.random.choice(allowed_actions)
        obs, rewards, done, truncated, info = env.step(action)
        episode_rewards += rewards
    return episode_rewards

def greedy_action(env):
    """Select the legal action with the highest immediate reward."""
    best_action = None
    best_reward = -np.inf
    for action in range(env.action_space.n):
        if env.allowed_actions[action]:
            if env.current_node is None:
                reward = -float(env.cost_matrix[action, action])  # Cost for selecting the first node
            else:
                reward = -float(env.distance_matrix[env.current_node, action])
            if reward > best_reward:
                best_reward = reward
                best_action = action
    return best_action

# Load the trained Maskable PPO model
model = MaskablePPO.load("p05_matrix_25_new", verbose=0)

# Values of p to evaluate
p_values = [0,0.01, 0.05, 0.1, 0.2, 0.5]

# Initialize lists to store average rewards and confidence intervals for each method
average_agent_rewards = []
average_greedy_rewards = []
average_random_rewards = []
agent_cis = []
greedy_cis = []
random_cis = []

# Evaluate for each value of p
for p in p_values:
    print(f"Evaluating for p = {p}...")
    env = CustomEnv(25, p=p)

    agent_rewards = []
    greedy_rewards = []
    random_rewards = []

    # Generate and solve 100 instances
    for instance in range(100):
        obs, _ = env.reset(fixed_instance=False)

        # Solve using the agent (deterministic)
        done = False
        truncated = False
        episode_rewards = 0
        while not (done or truncated):
            action_masks = env.action_masks()
            action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, rewards, done, truncated, info = env.step(action)
            episode_rewards += rewards
        agent_rewards.append(episode_rewards)

        # Solve using the greedy algorithm
        obs, _ = env.reset(fixed_instance=True)
        done = False
        truncated = False
        episode_rewards = 0
        while not (done or truncated):
            action = greedy_action(env)
            obs, rewards, done, truncated, info = env.step(action)
            episode_rewards += rewards
        greedy_rewards.append(episode_rewards)

        # Solve using the random algorithm
        random_reward = generate_random_solution(env)
        random_rewards.append(random_reward)

    # Calculate average rewards and confidence intervals for this value of p
    average_agent_rewards.append(np.mean(agent_rewards))
    average_greedy_rewards.append(np.mean(greedy_rewards))
    average_random_rewards.append(np.mean(random_rewards))

    # Calculate standard error of the mean (SEM) and confidence intervals
    agent_cis.append(1.96 * np.std(agent_rewards) / np.sqrt(len(agent_rewards)))
    greedy_cis.append(1.96 * np.std(greedy_rewards) / np.sqrt(len(greedy_rewards)))
    random_cis.append(1.96 * np.std(random_rewards) / np.sqrt(len(random_rewards)))

# Plot the results with confidence intervals
plt.figure(figsize=(10, 6))

# Agent
plt.plot(p_values, average_agent_rewards, label="Agent", marker="o", color="blue")
plt.fill_between(p_values,
                 np.array(average_agent_rewards) - np.array(agent_cis),
                 np.array(average_agent_rewards) + np.array(agent_cis),
                 color="blue", alpha=0.2)

# Greedy
plt.plot(p_values, average_greedy_rewards, label="Greedy", marker="s", color="green")
plt.fill_between(p_values,
                 np.array(average_greedy_rewards) - np.array(greedy_cis),
                 np.array(average_greedy_rewards) + np.array(greedy_cis),
                 color="green", alpha=0.2)

# Random
plt.plot(p_values, average_random_rewards, label="Random", marker="^", color="orange")
plt.fill_between(p_values,
                 np.array(average_random_rewards) - np.array(random_cis),
                 np.array(average_random_rewards) + np.array(random_cis),
                 color="orange", alpha=0.2)

plt.xlabel("p (Precedence Constraint Probability)")
plt.ylabel("Average Reward")
plt.title("Average Rewards for Different Values of p (with Confidence Intervals)")
plt.legend()
plt.grid()
plt.ylim(None, 0)  # Set the lower limit of the y-axis to 0, and let the upper limit adjust automatically
plt.tight_layout()
plt.show()