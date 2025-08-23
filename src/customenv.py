import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matrices import generate_random_matrices, generate_no_wait_flow_shop_instance, generate_clustered_matrices, generate_approximate_shortest_common_superstring_instances, generate_euclidic_matrices, generate_clustered_matrices_with_random_asymmetry
import matplotlib.pyplot as plt
import networkx as nx

import time
from copy import deepcopy

class CustomEnv(gym.Env):
    "Custom environment for the asymmetric TSP with precedence constraints."
    
    def __init__(self, N_GRAPH_SIZE,type = "Random", normalize_rewards = False, **kwargs):
        super().__init__()
        self.N_GRAPH_SIZE = N_GRAPH_SIZE
        self.type = type  # Type of the problem instance (Random, Clustered, FlowShop, StringDistance
        # Generate matrices
        #self.normalize_rewards = normalize_rewards  # Flag to normalize rewards
        self.kwargs = kwargs  # Store additional keyword arguments for matrix generation
        match type:
            case "Random":
                self.distance_matrix, self.precedence_matrix, self.reduced_precedence_matrix, self.cost_matrix = generate_random_matrices(N_GRAPH_SIZE,**kwargs)
            case "Clustered":
                self.distance_matrix, self.precedence_matrix, self.reduced_precedence_matrix, self.cost_matrix = generate_clustered_matrices(N_GRAPH_SIZE, **kwargs)
            case "FlowShop":
                self.distance_matrix, self.precedence_matrix, self.reduced_precedence_matrix, self.cost_matrix = generate_no_wait_flow_shop_instance(N_GRAPH_SIZE, **kwargs)
            case "StringDistance":
                self.distance_matrix, self.precedence_matrix, self.reduced_precedence_matrix, self.cost_matrix = generate_approximate_shortest_common_superstring_instances(N_GRAPH_SIZE,**kwargs)
            case "Euclidic":
                self.distance_matrix, self.precedence_matrix, self.reduced_precedence_matrix, self.cost_matrix = generate_euclidic_matrices(N_GRAPH_SIZE,**kwargs)
            case "ClusteredWithRandomAsymmetry":
                    self.distance_matrix, self.precedence_matrix, self.reduced_precedence_matrix, self.cost_matrix = generate_clustered_matrices_with_random_asymmetry(self.N_GRAPH_SIZE, **self.kwargs)


        # Define action and observation space
        self.action_space = spaces.Discrete(N_GRAPH_SIZE)  # there are N_GRAPH_SIZE possible actions
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_GRAPH_SIZE, N_GRAPH_SIZE, 4), dtype=np.uint8)
        
        self.visited_nodes = np.zeros(N_GRAPH_SIZE, dtype=bool)
        self.allowed_actions = np.ones(N_GRAPH_SIZE, dtype=bool)
        self.current_node = None  # No current node at the start
        self.actions_taken = []
        self._set_allowed_actions()  # Set allowed actions based on initial state
        self.max_value = max(np.max(self.distance_matrix), np.max(self.cost_matrix),0.001)
        self.distance_matrix = self.distance_matrix / self.max_value
        self.cost_matrix = self.cost_matrix / self.max_value


        self.original_cost_matrix = np.diag(self.cost_matrix).copy()  # Save the original cost matrix


        self._observation_buffer = np.zeros((N_GRAPH_SIZE,N_GRAPH_SIZE,4), dtype=np.float32)  # Buffer for observations
        self._observation_buffer[:, :, 0] = np.where(np.transpose(self.precedence_matrix) ==1,0,self.distance_matrix) #set weights of illegal edges to 0
        self._observation_buffer[:, :, 1] = self.precedence_matrix
        #self.reward_history = [] this was used to normalize rewards, but since I already normalize the distance matrix, I am currently not normalizing the rewards
        #self.reward_mean = 0.0
        #self.reward_std = 1.0
        #self.update_freq = 100

    def step(self, action):
        if not self.allowed_actions[action]:
            reward = -100.0 + (-10.0 * np.sum(~self.visited_nodes))  # Penalty for violating precedence constraints
            terminated = True
            truncated = True
        else:
            if self.current_node is None:
                reward = -float(self.cost_matrix[action, action])  # Cost for selecting the first node
            else:
                reward = -float(self.distance_matrix[self.current_node, action])
            
            self.visited_nodes[action] = True
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

        self._set_allowed_actions()  # Update allowed actions based on the current state
        observation = self._get_observation()

        #self.reward_history.append(reward)
        #if len(self.reward_history) >= self.update_freq:
         #   self.reward_mean = np.mean(self.reward_history)
          #  self.reward_std = np.std(self.reward_history)
           # self.reward_history = []
        #if hasattr(self, 'normalize_rewards') and self.normalize_rewards:
         #   reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
          #  reward = np.clip(reward, -50, 50)

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None,fixed_instance = False,shuffle = False):
        # Regenerate matrices to sample a new problem
        if not fixed_instance:
            match self.type:
                case "Random":
                    self.distance_matrix, self.precedence_matrix, self.reduced_precedence_matrix, self.cost_matrix = generate_random_matrices(self.N_GRAPH_SIZE,**self.kwargs)
                case "Clustered":
                    self.distance_matrix, self.precedence_matrix, self.reduced_precedence_matrix, self.cost_matrix = generate_clustered_matrices(self.N_GRAPH_SIZE, **self.kwargs)
                case "FlowShop":
                    self.distance_matrix, self.precedence_matrix, self.reduced_precedence_matrix, self.cost_matrix = generate_no_wait_flow_shop_instance(self.N_GRAPH_SIZE, **self.kwargs)
                case "StringDistance":
                    self.distance_matrix, self.precedence_matrix, self.reduced_precedence_matrix, self.cost_matrix = generate_approximate_shortest_common_superstring_instances(self.N_GRAPH_SIZE,**self.kwargs)
                case "Euclidic":
                    self.distance_matrix, self.precedence_matrix, self.reduced_precedence_matrix, self.cost_matrix = generate_euclidic_matrices(self.N_GRAPH_SIZE, **self.kwargs)
                case "ClusteredWithRandomAsymmetry":
                    self.distance_matrix, self.precedence_matrix, self.reduced_precedence_matrix, self.cost_matrix = generate_clustered_matrices_with_random_asymmetry(self.N_GRAPH_SIZE, **self.kwargs)

            self.max_value = max(np.max(self.distance_matrix), np.max(self.cost_matrix),0.0001)
            self.distance_matrix = self.distance_matrix / self.max_value
            self.cost_matrix = self.cost_matrix / self.max_value
            self._observation_buffer[:, :, 0] = np.where(np.transpose(self.precedence_matrix) ==1,0,self.distance_matrix) #set weights of illegal edges to 0
            self._observation_buffer[:, :, 1] = self.precedence_matrix
            self.original_cost_matrix = np.diag(self.cost_matrix).copy()  # Save the original cost matrix

        else:
            self.cost_matrix = np.diag(self.original_cost_matrix.copy()) #reset the cost matrix to the original cost matrix

            if shuffle:
                # Shuffle the precedence matrix and cost matrix, this is to compare the performance of the agent of the same instance but with a the nodes in a different order
                indices = np.random.permutation(self.N_GRAPH_SIZE)
                self.distance_matrix = self.distance_matrix[indices][:, indices]
                self.precedence_matrix = self.precedence_matrix[indices][:, indices]
                self.reduced_precedence_matrix = self.reduced_precedence_matrix[indices][:, indices]
                self.cost_matrix = self.cost_matrix[indices][:, indices]
                self.original_cost_matrix = self.original_cost_matrix[indices]
                self._observation_buffer[:, :, 0] = np.where(np.transpose(self.precedence_matrix) ==1,0,self.distance_matrix) #set weights of illegal edges to 0
                self._observation_buffer[:, :, 1] = self.precedence_matrix
        self.visited_nodes.fill(False)
        self.current_node = None
        self.actions_taken = []  # Reset actions taken
        self._set_allowed_actions()  # Set allowed actions based on the initial state
        self._get_observation()
        return self._observation_buffer, {}

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
        self._observation_buffer[:, :, 2] = self.cost_matrix
        self._observation_buffer[:, :, 3] = self.visited_nodes[:, np.newaxis]  # Add visited nodes as the fourth channel
        return self._observation_buffer

    def _check_precedence_constraints(self, action):
        # Check if all nodes that have precedence constraints for the current action have been visited
        predecessors = (self.precedence_matrix[~self.visited_nodes, action] == 1)
        return not np.any(predecessors) and not self.visited_nodes[action]  # Action is allowed if no predecessors are unvisited and the node itself is not visited
    
    def _set_allowed_actions(self):
        # Start with assuming all unvisited nodes are allowed
        not_visited = ~self.visited_nodes
        self.allowed_actions = not_visited.copy()
        
        # Disable current node if it exists
        if self.current_node is not None:
            self.allowed_actions[self.current_node] = False
            
        # For each unvisited node, check if its prerequisites are all visited
        for node in np.where(not_visited)[0]:
            # Get all nodes that must come before this one
            prerequisite_nodes = np.where(self.precedence_matrix[:, node] == 1)[0]
            
            # If any prerequisite is unvisited, this action isn't allowed
            if np.any(~self.visited_nodes[prerequisite_nodes]):
                self.allowed_actions[node] = False

    def print_precedence_constraints(self):
        print("Precedence Constraints:")
        for i in range(self.N_GRAPH_SIZE):
            for j in range(self.N_GRAPH_SIZE):
                if self.reduced_precedence_matrix[i, j] == 1:
                       print(f"{i} -> {j}")

    def check_tour(self,tour = None,normalize = True):
        "Check if the tour is valid and returdn the total cost."

        if not tour:
            tour = self.actions_taken

        if len(tour) != self.N_GRAPH_SIZE:
            print("Tour is not complete. Not all nodes were visited.")
            return None
    

        for i in range(self.N_GRAPH_SIZE):
            for j in range(self.N_GRAPH_SIZE):
                if self.reduced_precedence_matrix[i, j] == 1:
                    # Node i must come before node j
                    pos_i = tour.index(i)
                    pos_j = tour.index(j)
                    if pos_i >= pos_j:
                        return None, False, f"Precedence constraint violated: {i} must come before {j}"
    
        # Calculate tour cost
        total_cost = 0.0

        if normalize:
            max_value = 1
        else:
            max_value = self.max_value
    
        # Add cost of first node
        total_cost += float(self.original_cost_matrix[tour[0]] * max_value)
    
        # Add costs of transitions
        for i in range(len(tour) - 1):
            from_node = tour[i]
            to_node = tour[i + 1]
            total_cost += float(self.distance_matrix[from_node, to_node] * max_value)
    
        return total_cost, True, "Valid tour"

    def save_state(self):
        """
        Save the current state of the environment.
        
        Returns:
            dict: A dictionary containing all necessary state information
        """
        state = {
            # Graph and problem definition
            'distance_matrix': self.distance_matrix.copy(),
            'precedence_matrix': self.precedence_matrix.copy() if hasattr(self, 'precedence_matrix') else None,
            'reduced_precedence_matrix': self.reduced_precedence_matrix.copy() if hasattr(self, 'reduced_precedence_matrix') else None,
            'cost_matrix': self.cost_matrix.copy() if hasattr(self, 'cost_matrix') else None,
            
            # Current state
            'current_node': self.current_node,
            'visited_nodes': self.visited_nodes.copy(),
            'allowed_actions': self.allowed_actions.copy(),
            'history': self.history.copy() if hasattr(self, 'history') else [],
            
            # Episode tracking

            'total_reward': self.total_reward if hasattr(self, 'total_reward') else 0,
            'done': self.done if hasattr(self, 'done') else False,
            'truncated': self.truncated if hasattr(self, 'truncated') else False,
            
            
            # Additional environment parameters
            'graph_size': self.N_GRAPH_SIZE,
            'type': self.type,
            'normalize_rewards': self.normalize_rewards if hasattr(self, 'normalize_rewards') else True,
            'p': self.p if hasattr(self, 'p') else 0,
            'seed': self.seed_value if hasattr(self, 'seed_value') else None,
            
            # Add these missing variables:
            'actions_taken': self.actions_taken.copy(),
            '_observation_buffer': self._observation_buffer.copy(),
            'original_cost_matrix': self.original_cost_matrix.copy(),
            'max_value': self.max_value,
        }
        
        return state

    def load_state(self, state):
        """
        Restore the environment to a previously saved state.
        
        Args:
            state (dict): A state dictionary returned by save_state()
            
        Returns:
            observation: The observation corresponding to the restored state
        """
        # Restore graph and problem definition
        self.distance_matrix = state['distance_matrix'].copy()
        
        if state['precedence_matrix'] is not None:
            self.precedence_matrix = state['precedence_matrix'].copy()
        
        if state['reduced_precedence_matrix'] is not None:
            self.reduced_precedence_matrix = state['reduced_precedence_matrix'].copy()


        self.cost_matrix = state['cost_matrix'].copy() if 'cost_matrix' in state else np.diag(self.distance_matrix).copy()
        
        # Restore current state
        self.current_node = state['current_node']
        self.visited_nodes = state['visited_nodes'].copy()
        self.allowed_actions = state['allowed_actions'].copy()
        
        if 'history' in state:
            self.history = state['history'].copy()
        
        # Restore episode tracking
        
        if 'total_reward' in state:
            self.total_reward = state['total_reward']
        
        if 'done' in state:
            self.done = state['done']
        
        if 'truncated' in state:
            self.truncated = state['truncated']
        
        # Add these missing restores:
        if 'actions_taken' in state:
            self.actions_taken = state['actions_taken'].copy()
        
        if '_observation_buffer' in state:
            self._observation_buffer = state['_observation_buffer'].copy()
        
        if 'original_cost_matrix' in state:
            self.original_cost_matrix = state['original_cost_matrix'].copy()
        
        if 'max_value' in state:
            self.max_value = state['max_value']
        
        # Generate observation based on current state
        observation = self._get_observation()
        
        return observation




