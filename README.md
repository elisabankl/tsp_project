# tsp-project/tsp-project/README.md

# Asymmetric Traveling Salesman Problem with Precedence Constraints

This project implements a custom environment for solving the Asymmetric Traveling Salesman Problem (TSP) with precedence constraints using reinforcement learning techniques. The environment is designed to work with Stable Baselines algorithms and incorporates unique features such as cost penalties for node selection and strict adherence to precedence constraints.

## Overview

The asymmetric TSP is a variation of the classic TSP where the distance from node A to node B may differ from the distance from node B to node A. In this project, we introduce precedence constraints that dictate the order in which nodes must be visited. The goal is to visit each node exactly once without completing a tour, while minimizing the total cost associated with the path taken.

## Project Structure

- **src/customenv.py**: Defines the `CustomEnv` class, which manages the action and observation spaces, implements the step, reset, and render methods, and integrates with Stable Baselines algorithms.
  
- **src/matrices.py**: Contains functions to generate the necessary matrices for the environment:
  1. An asymmetric distance matrix defining the distances between nodes.
  2. A binary precedence constraint matrix indicating which nodes have precedence constraints.
  3. A diagonal cost matrix specifying the cost of selecting each node as the first node.

- **src/types/index.py**: Defines custom types or data structures needed for the project, such as classes or interfaces for representing nodes, actions, and states.

- **requirements.txt**: Lists the dependencies required for the project, including `gymnasium`, `numpy`, and `stable-baselines3`.

- **setup.py**: Used for packaging the project, including metadata and specifying dependencies.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd tsp-project
pip install -r requirements.txt
```

## Usage

To use the custom environment, you can create an instance of `CustomEnv` and interact with it using Stable Baselines algorithms. Here is a simple example:

```python
from src.customenv import CustomEnv

env = CustomEnv(N_GRAPH_SIZE)
# Interact with the environment using Stable Baselines algorithms
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.