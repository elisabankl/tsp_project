import numpy as np
from sb3_contrib import MaskablePPO
from customenv import CustomEnv
import pandas as pd
import scipy.stats as stats
from datetime import datetime
from or_tools_google import solve_google_or_with_greedy_solution
from compare_permutate import solve_tsp_cheapest_insertion
import json
import os
import uuid
import time
from collections import defaultdict


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



# Simple lightweight timing helper
class TimerCollection:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)

    class _Timer:
        def __init__(self, coll, key):
            self.coll = coll
            self.key = key
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            self.coll.times[self.key] += dt
            self.coll.counts[self.key] += 1

    def timeit(self, key):
        return TimerCollection._Timer(self, key)

def evaluate_multiple_agents(agent_paths, p_values, instance_type="Random", graph_size=25, n_instances=100, n_permutations = 10, ant_solver_enabled=True, or_tools_enabled=True, n_nondeterministic_runs=10, profiling_enabled=False, profiling_output_dir="results2/profiling", **instance_kwargs):
    """
    Evaluate multiple agents on the same set of instances.

    profiling_enabled: if True collect timing statistics and write summary JSON per-run
    profiling_output_dir: directory to store profiling summaries
    """

    results_root = os.path.join("results2", instance_type)
    os.makedirs(results_root, exist_ok=True)

    # Load all agents
    agents = []
    agent_names = []
    for path in agent_paths:
        model = MaskablePPO.load(path, verbose=0)
        agents.append(model)
        # Extract agent name from path
        agent_name = path.split("/")[-1].split(".")[0]
        agent_names.append(agent_name)
    
    # Initialize Ant Colony solver with 1 second timeout
    if ant_solver_enabled:
        try:
            from sop_solver import SOPSolver
            sop_solver = SOPSolver(timeout=1)
            print("Ant Colony Solver (SOP-ACS) initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Ant Colony Solver: {e}")
            ant_solver_enabled = False
 
    # Data structures to store results for each agent
        # Add non-deterministic version of each agent
    
    # profiling setup
    timers = TimerCollection() if profiling_enabled else None
    if profiling_enabled:
        os.makedirs(profiling_output_dir, exist_ok=True)
        run_start = time.perf_counter()

    # Evaluate for each value of p
    for p in p_values:
        print(f"Evaluating for p = {p}...")
        p_dir = os.path.join(results_root, f"p_{p}")
        os.makedirs(p_dir, exist_ok=True)

        for instance_idx in range(n_instances):
            if profiling_enabled:
                with timers.timeit("env_create"):
                    env = CustomEnv(graph_size, instance_type, p=p, **instance_kwargs)
                    obs, _ = env.reset(fixed_instance=False)
            else:
                env = CustomEnv(graph_size, instance_type, p=p, **instance_kwargs)
                obs, _ = env.reset(fixed_instance=False)

            if profiling_enabled:
                with timers.timeit("env_save_state"):
                    env_state = env.save_state()
            else:
                env_state = env.save_state()

            instance_id = f"{instance_idx:04d}_{uuid.uuid4().hex[:8]}"
            timestamp = datetime.now().isoformat()

            # collect matrices (convert to python lists)
            if profiling_enabled:
                with timers.timeit("matrix_extract"):
                    distance_matrix = env.distance_matrix.tolist()
                    precedence_matrix = env.precedence_matrix.tolist()
                    cost_matrix = (env.cost_matrix.tolist() if hasattr(env, "cost_matrix")
                                   else np.diag(env.original_cost_matrix).tolist())
            else:
                distance_matrix = env.distance_matrix.tolist()
                precedence_matrix = env.precedence_matrix.tolist()
                cost_matrix = (env.cost_matrix.tolist() if hasattr(env, "cost_matrix")
                               else np.diag(env.original_cost_matrix).tolist())

            solvers_results = {}

            # OR-Tools
            if or_tools_enabled:
                if profiling_enabled:
                    with timers.timeit("ortools_total"):
                        try:
                            solution, route = solve_google_or_with_greedy_solution(
                                env.distance_matrix, np.diag(env.original_cost_matrix), env.reduced_precedence_matrix, timeout=1
                            )
                            if route:
                                env.load_state(env_state)
                                or_reward, _, _ = env.check_tour(route)
                                solvers_results['OR-Tools'] = {
                                    "reward": float(-or_reward) if or_reward is not None else None,
                                    "tour": list(route) if route is not None else None
                                }
                            else:
                                solvers_results['OR-Tools'] = {"reward": None, "tour": None}
                        except Exception:
                            solvers_results['OR-Tools'] = {"reward": None, "tour": None}
                else:
                    try:
                        solution, route = solve_google_or_with_greedy_solution(
                            env.distance_matrix, np.diag(env.original_cost_matrix), env.reduced_precedence_matrix, timeout=1
                        )
                        if route:
                            env.load_state(env_state)
                            or_reward, _, _ = env.check_tour(route)
                            solvers_results['OR-Tools'] = {
                                "reward": float(-or_reward) if or_reward is not None else None,
                                "tour": list(route) if route is not None else None
                            }
                        else:
                            solvers_results['OR-Tools'] = {"reward": None, "tour": None}
                    except Exception:
                        solvers_results['OR-Tools'] = {"reward": None, "tour": None}

            # AntColony
            if ant_solver_enabled:
                if profiling_enabled:
                    with timers.timeit("ant_total"):
                        try:
                            solution = sop_solver.solve_from_matrices(
                                distance_matrix=env.distance_matrix,
                                precedence_matrix=env.precedence_matrix,
                                cost_matrix=np.diag(env.cost_matrix) if hasattr(env, "cost_matrix") else np.zeros(graph_size),
                                instance_name=f"p{p}_inst{instance_idx}"
                            )
                            if solution and "runs" in solution and solution["runs"]:
                                path = [x-1 for x in solution["runs"][0]["best_solution"]][1:-1]
                                env.load_state(env_state)
                                ant_reward, _, _ = env.check_tour(path)
                                solvers_results['AntColony'] = {"reward": float(-ant_reward) if ant_reward is not None else None, "tour": list(path)}
                            else:
                                solvers_results['AntColony'] = {"reward": None, "tour": None}
                        except Exception:
                            solvers_results['AntColony'] = {"reward": None, "tour": None}
                else:
                    try:
                        solution = sop_solver.solve_from_matrices(
                            distance_matrix=env.distance_matrix,
                            precedence_matrix=env.precedence_matrix,
                            cost_matrix=np.diag(env.cost_matrix) if hasattr(env, "cost_matrix") else np.zeros(graph_size),
                            instance_name=f"p{p}_inst{instance_idx}"
                        )
                        if solution and "runs" in solution and solution["runs"]:
                            path = [x-1 for x in solution["runs"][0]["best_solution"]][1:-1]
                            env.load_state(env_state)
                            ant_reward, _, _ = env.check_tour(path)
                            solvers_results['AntColony'] = {"reward": float(-ant_reward) if ant_reward is not None else None, "tour": list(path)}
                        else:
                            solvers_results['AntColony'] = {"reward": None, "tour": None}
                    except Exception:
                        solvers_results['AntColony'] = {"reward": None, "tour": None}

            # NearestNeighbor (deterministic)
            if profiling_enabled:
                with timers.timeit("nn_run"):
                    env.load_state(env_state)
                    env._get_observation()
                    done = False
                    truncated = False
                    nn_reward = 0.0
                    nn_tour = []
                    while not (done or truncated):
                        a = greedy_action(env)
                        nn_tour.append(int(a))
                        _, r, done, truncated, _ = env.step(a)
                        nn_reward += r
            else:
                env.load_state(env_state)
                env._get_observation()
                done = False
                truncated = False
                nn_reward = 0.0
                nn_tour = []
                while not (done or truncated):
                    a = greedy_action(env)
                    nn_tour.append(int(a))
                    _, r, done, truncated, _ = env.step(a)
                    nn_reward += r

            solvers_results['NearestNeighbor'] = {"reward": float(nn_reward), "tour": nn_tour}

            # Cheapest Insertion
            if profiling_enabled:
                with timers.timeit("ci_run"):
                    try:
                        ni_solution, ci_reward = solve_tsp_cheapest_insertion(env.distance_matrix, env.precedence_matrix,env.original_cost_matrix)
                        solvers_results['Cheapest Insertion'] = {"reward": float(ci_reward), "tour": list(ni_solution) if ni_solution is not None else None}
                    except Exception:
                        solvers_results['Cheapest Insertion'] = {"reward": None, "tour": None}
            else:
                try:
                    ni_solution, ci_reward = solve_tsp_cheapest_insertion(env.distance_matrix, env.precedence_matrix,env.original_cost_matrix)
                    solvers_results['Cheapest Insertion'] = {"reward": float(ci_reward), "tour": list(ni_solution) if ni_solution is not None else None}
                except Exception:
                    solvers_results['Cheapest Insertion'] = {"reward": None, "tour": None}

            # Agents: deterministic, nondet runs, permutations
            for agent_idx, agent_name in enumerate(agent_names):
                agent = agents[agent_idx]

                # deterministic run (record tour)
                if profiling_enabled:
                    with timers.timeit("agent_det_run"):
                        env.load_state(env_state)
                        obs = env._get_observation()
                        done = False
                        truncated = False
                        det_reward = 0.0
                        det_tour = []
                        greedy_count = 0
                        total_actions = 0
                        while not (done or truncated):
                            masks = env.action_masks()
                            action, _ = agent.predict(obs, deterministic=True, action_masks=masks)
                            det_tour.append(int(action))
                            if action == greedy_action(env):
                                greedy_count += 1
                            obs, r, done, truncated, _ = env.step(action)
                            det_reward += r
                            total_actions += 1
                        greedy_pct = greedy_count
                else:
                    env.load_state(env_state)
                    obs = env._get_observation()
                    done = False
                    truncated = False
                    det_reward = 0.0
                    det_tour = []
                    greedy_count = 0
                    total_actions = 0
                    while not (done or truncated):
                        masks = env.action_masks()
                        action, _ = agent.predict(obs, deterministic=True, action_masks=masks)
                        det_tour.append(int(action))
                        if action == greedy_action(env):
                            greedy_count += 1
                        obs, r, done, truncated, _ = env.step(action)
                        det_reward += r
                        total_actions += 1
                    greedy_pct = greedy_count

                # nondeterministic runs (store rewards and greedy% per run)
                nondet_rewards = []
                nondet_greedy_pct = []
                if profiling_enabled:
                    with timers.timeit("agent_nondet_all"):
                        for run in range(n_nondeterministic_runs):
                            env.load_state(env_state)
                            obs = env._get_observation()
                            done = False
                            truncated = False
                            run_reward = 0.0
                            run_greedy = 0
                            run_actions = 0
                            while not (done or truncated):
                                masks = env.action_masks()
                                action, _ = agent.predict(obs, deterministic=False, action_masks=masks)
                                if action == greedy_action(env):
                                    run_greedy += 1
                                obs, r, done, truncated, _ = env.step(action)
                                run_reward += r
                                run_actions += 1
                            nondet_rewards.append(float(run_reward))
                            nondet_greedy_pct.append(run_greedy)
                else:
                    for run in range(n_nondeterministic_runs):
                        env.load_state(env_state)
                        obs = env._get_observation()
                        done = False
                        truncated = False
                        run_reward = 0.0
                        run_greedy = 0
                        run_actions = 0
                        while not (done or truncated):
                            masks = env.action_masks()
                            action, _ = agent.predict(obs, deterministic=False, action_masks=masks)
                            if action == greedy_action(env):
                                run_greedy += 1
                            obs, r, done, truncated, _ = env.step(action)
                            run_reward += r
                            run_actions += 1
                        nondet_rewards.append(float(run_reward))
                        nondet_greedy_pct.append(run_greedy)

                # permutations (only rewards + greedy%)
                perm_rewards = []
                perm_greedy_pct = []
                if profiling_enabled:
                    with timers.timeit("agent_perm_all"):
                        for pid in range(n_permutations):
                            env.reset(fixed_instance=True)
                            env.load_state(env_state)
                            env.reset(fixed_instance=True, shuffle=True)
                            obs = env._get_observation()
                            done = False
                            truncated = False
                            p_reward = 0.0
                            p_greedy = 0
                            p_actions = 0
                            while not (done or truncated):
                                masks = env.action_masks()
                                action, _ = agent.predict(obs, deterministic=True, action_masks=masks)
                                if action == greedy_action(env):
                                    p_greedy += 1
                                obs, r, done, truncated, _ = env.step(action)
                                p_reward += r
                                p_actions += 1
                            perm_rewards.append(float(p_reward))
                            perm_greedy_pct.append(p_greedy)
                else:
                    for pid in range(n_permutations):
                        env.reset(fixed_instance=True)
                        env.load_state(env_state)
                        env.reset(fixed_instance=True, shuffle=True)
                        obs = env._get_observation()
                        done = False
                        truncated = False
                        p_reward = 0.0
                        p_greedy = 0
                        p_actions = 0
                        while not (done or truncated):
                            masks = env.action_masks()
                            action, _ = agent.predict(obs, deterministic=True, action_masks=masks)
                            if action == greedy_action(env):
                                p_greedy += 1
                            obs, r, done, truncated, _ = env.step(action)
                            p_reward += r
                            p_actions += 1
                        perm_rewards.append(float(p_reward))
                        perm_greedy_pct.append(p_greedy)
                solvers_results[agent_name] = {
                    "reward": float(det_reward),
                    "tour": det_tour,
                    "nn_choice_percentage": float(greedy_pct),
                    "nondet_runs": {
                        "rewards": nondet_rewards,
                        "nn_percentages": nondet_greedy_pct
                    },
                    "permutations": {
                        "rewards": perm_rewards,
                        "nn_percentages": perm_greedy_pct
                    }
                }

            # assemble instance record
            instance_record = {
                "instance_id": instance_id,
                "p_value": float(p),
                "instance_index": int(instance_idx),
                "graph_size": int(graph_size),
                "instance_type": instance_type,
                "timestamp": timestamp,
                "matrices": {
                    "distance_matrix": distance_matrix,
                    "precedence_matrix": precedence_matrix,
                    "cost_matrix": cost_matrix
                },
                "solvers": solvers_results,
                "metadata": {
                    "env_kwargs": instance_kwargs
                }
            }

            # write JSON atomically
            if profiling_enabled:
                with timers.timeit("json_dump"):
                    out_path = os.path.join(p_dir, f"instance_{instance_id}.json")
                    tmp_path = out_path + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(instance_record, f, indent=2)
                    os.replace(tmp_path, out_path)
            else:
                out_path = os.path.join(p_dir, f"instance_{instance_id}.json")
                tmp_path = out_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(instance_record, f, indent=2)
                os.replace(tmp_path, out_path)
    # end per-p and instances loops

    # write profiling summary if enabled
    if profiling_enabled:
        total_run = time.perf_counter() - run_start
        profile_summary = {
            "total_run_time_s": total_run,
            "timers": {k: float(v) for k, v in timers.times.items()},
            "counts": {k: int(v) for k, v in timers.counts.items()}
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_path = os.path.join(profiling_output_dir, f"profile_{instance_type}_{ts}.json")
        with open(profile_path, "w", encoding="utf-8") as pf:
            json.dump(profile_summary, pf, indent=2)
        print(f"Profiling summary written to {profile_path}")


    

def evaluate_agents(instance_type,agent_paths, ant_solver_enabled=True,or_solver_enabled=True, **instance_kwargs):
    """Main function to evaluate and compare multiple agents."""
    # List of agent model paths to compare
    
    
    # Values of p to evaluate
    p_values = [0,0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,0.125,0.15,0.2,0.3]
    
    n_instances = 100  # Reduced for testing, increase for production
    graph_size = 25
    
    # Run evaluation with all agents and solvers
    evaluate_multiple_agents(
        agent_paths=agent_paths,
        p_values=p_values,
        instance_type=instance_type,
        ant_solver_enabled=ant_solver_enabled,
        or_tools_enabled=or_solver_enabled,
        graph_size=graph_size,
        n_instances=n_instances,
        **instance_kwargs
    )
    
    print(f"Evaluation for {instance_type} complete")

if __name__ == "__main__":
    # Everything is now properly contained in functions
    # Only the evaluate_agents function is called here
    agent_paths = [
        "25_flowshop_masked_ppo_20250628_130538.zip",
        "25_euclidic_masked_ppo_20250623_021838.zip",
        "25_random_masked_ppo_20250626_100225.zip",
        "25_stringdistance_masked_ppo_20250629_130536.zip",  
        "25_euclidic_masked_ppo_20250623_021838.zip" 
         ]
    instance_types = ["FlowShop","Euclidic","Random","StringDistance","ClusteredWithRandomAsymmetry"]
    
    instance_type = "FlowShop" #instance_types[i]
    agent_path = ["25_flowshop_masked_ppo_20250628_130538.zip"] #,"25_clusteredwithrandomasymmetry_masked_ppo_20250805_222638.zip"]
    print(f"Evaluating agent {agent_path} for instance type: {instance_type}")
    #evaluate_agents(instance_type=instance_type,agent_paths = agent_path, or_solver_enabled=True,ant_solver_enabled=True,profiling_enabled = True, n_jobs=50)
    #evaluate_agent("25_clusteredwithrandomasymmetry_masked_ppo_20250714_230351",instance_type="ClusteredWithRandomAsymmetry", graph_size=25)
    #evaluate_agents(instance_type="StringDistance",agent_paths = ["25_stringdistance_masked_ppo_20250629_130536.zip"], or_solver_enabled=True,ant_solver_enabled=True)
    #evaluate_agents(instance_type="Random",agent_paths = ["25_random_masked_ppo_20250626_100225.zip"], or_solver_enabled=True,ant_solver_enabled=True)
    #evaluate_agents(instance_type="Euclidic",agent_paths = ["25_euclidic_masked_ppo_20250623_021838.zip"], or_solver_enabled=True,ant_solver_enabled=True)
    #evaluate_agents(instance_type="ClusteredWithRandomAsymmetry",agent_paths = ["25_euclidic_masked_ppo_20250623_021838.zip"], or_solver_enabled=True,ant_solver_enabled=True)
    evaluate_agents(instance_type="FlowShop",agent_paths = ["25_flowshop_masked_ppo_20250628_130538.zip"], or_solver_enabled=True,ant_solver_enabled=True)


