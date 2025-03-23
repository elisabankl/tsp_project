import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matrices import generate_distance_matrix, generate_precedence_matrix, generate_cost_matrix
import matplotlib.pyplot as plt
import networkx as nx

class CustomEnv(gym.Env):
    "Custom environment for the asymmetric TSP with precedence constraints."
    
    def __init__(self, N_GRAPH_SIZE):
        super().__init__()
        self.N_GRAPH_SIZE = N_GRAPH_SIZE
        
        # Generate matrices
        self.distance_matrix = generate_distance_matrix(N_GRAPH_SIZE)
        self.precedence_matrix = generate_precedence_matrix(N_GRAPH_SIZE)
        self.cost_matrix = generate_cost_matrix(N_GRAPH_SIZE)
        self.original_cost_matrix = np.diag(self.cost_matrix).copy()  # Save the original cost matrix
        
        # Define action and observation space
        self.action_space = spaces.Discrete(N_GRAPH_SIZE)  # there are N_GRAPH_SIZE possible actions
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_GRAPH_SIZE, N_GRAPH_SIZE, 4), dtype=np.uint8)
        
        self.visited_nodes = np.zeros(N_GRAPH_SIZE, dtype=np.uint8)
        self.current_node = None  # No current node at the start
        self.actions_taken = []  # Record actions taken
        self.allowed_actions = [not (self.visited_nodes[i] == 1) and self._check_precedence_constraints(i) for i in range(N_GRAPH_SIZE)]

    def step(self, action):
        if not self.allowed_actions[action]:
            reward = -100.0 + (-10.0 * np.sum(self.visited_nodes == 0))  # Penalty for violating precedence constraints
            terminated = True
            truncated = True
        else:
            if self.current_node is None:
                reward = -float(self.cost_matrix[action, action])  # Cost for selecting the first node
            else:
                reward = -float(self.distance_matrix[self.current_node, action])
            
            self.visited_nodes[action] = 1
            self.current_node = action
            self.actions_taken.append(int(action))  # Record the action as an integer
            
            # Check if all but one node has been visited
            if np.sum(self.visited_nodes) == self.N_GRAPH_SIZE - 1:
                last_node = np.where(self.visited_nodes == 0)[0][0]
                reward -= float(self.distance_matrix[self.current_node, last_node])
                terminated = True
                truncated = False
            else:
                np.fill_diagonal(self.cost_matrix, self.distance_matrix[self.current_node, :])  # Prevent visiting the same node again
                terminated = False
                truncated = False

        self.allowed_actions = [not (self.visited_nodes[i] == 1) and self._check_precedence_constraints(i) for i in range(self.N_GRAPH_SIZE)]
        observation = self._get_observation()
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None,fixed_instance = False):
        # Regenerate matrices to sample a new problem
        if not fixed_instance:
            self.distance_matrix = generate_distance_matrix(self.N_GRAPH_SIZE)
            self.precedence_matrix = generate_precedence_matrix(self.N_GRAPH_SIZE)
            self.cost_matrix = generate_cost_matrix(self.N_GRAPH_SIZE)
            self.original_cost_matrix = np.diag(self.cost_matrix).copy()  # Save the original cost matrix

        else:
            self.cost_matrix = np.diag(self.original_cost_matrix.copy())
        
        self.visited_nodes = np.zeros(self.N_GRAPH_SIZE, dtype=np.uint8)
        self.current_node = None
        self.actions_taken = []  # Reset actions taken
        self.allowed_actions = [not (self.visited_nodes[i] == 1) and self._check_precedence_constraints(i) for i in range(self.N_GRAPH_SIZE)]
        observation = self._get_observation()
        return observation, {}

    def render(self, mode='human'):
        # Create a directed graph with the distance matrix as edge weights and cost matrix as node weights
        G = nx.DiGraph()
        for i in range(self.N_GRAPH_SIZE):
            G.add_node(i, weight=self.original_cost_matrix[i])
            for j in range(self.N_GRAPH_SIZE):
                if self.distance_matrix[i, j] > 0 and not self.precedence_matrix[j, i] == 1:
                    G.add_edge(i, j, weight=self.distance_matrix[i, j])
                elif self.precedence_matrix[j, i] == 1:
                    G.add_edge(i, j, weight=-1)  # Edge j to i with weight -1 for precedence constraint
        
        pos = nx.spring_layout(G)
        node_labels = {i: f'{self.original_cost_matrix[i]:.2f}' for i in G.nodes}
        
        # Separate edge labels for each direction
        edge_labels_forward = {(i, j): f'{G[i][j]["weight"]:.2f}' for i, j in G.edges}
        edge_labels_backward = {(j, i): f'{G[j][i]["weight"]:.2f}' for i, j in G.edges}
        
        nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=500, font_size=10, font_weight='bold', arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_forward, font_size=8, label_pos=0.3, bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white'))
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_backward, font_size=8, label_pos=0.7, bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white'))
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

        # Highlight the path taken
        if self.actions_taken:
            path_edges = [(self.actions_taken[i], self.actions_taken[i + 1]) for i in range(len(self.actions_taken) - 1)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2, connectionstyle='arc3,rad=0.2', arrows=True)

        plt.title("TSP Path with Distances and Costs")
        plt.show()
    
    def action_masks(self):
        return self.allowed_actions

    def close(self):
        pass

    def _get_observation(self):
        # Create the observation matrix
        observation = np.zeros((self.N_GRAPH_SIZE, self.N_GRAPH_SIZE, 4), dtype=np.uint8)
        observation[:, :, 0] = self.distance_matrix
        observation[:, :, 1] = self.precedence_matrix
        observation[:, :, 2] = self.cost_matrix
        observation[:, :, 3] = self.visited_nodes[:, np.newaxis]  # Add visited nodes as the fourth channel
        return observation

    def _check_precedence_constraints(self, action):
        # Check if all nodes that have precedence constraints for the current action have been visited
        for i in range(self.N_GRAPH_SIZE):
            if self.precedence_matrix[i, action] == 1 and self.visited_nodes[i] == 0:
                return False
        return True



