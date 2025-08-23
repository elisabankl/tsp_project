import os
import json
import subprocess
import numpy as np
import tempfile
from datetime import datetime
import time
import glob

class SOPSolver:
    def __init__(self, container_name="acssa-builder", timeout=30):
        """Initialize the SOP solver with container configuration."""
        self.container_name = container_name
        self.timeout = timeout
        
        # Verify container exists - use more reliable method
        try:
            # First check using docker images command instead of inspect
            result = subprocess.run(
                ["docker", "images", "-q", container_name], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            if not result.stdout.strip():
                # Try with different format - sometimes tags matter
                result = subprocess.run(
                    ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                images = result.stdout.strip().split('\n')
                if not any(img.startswith(container_name) for img in images):
                    print(f"Container {container_name} not found. Please build it first.")
                    print("Run: docker build -t acssa-builder .")
                    raise RuntimeError(f"Container {container_name} not found")
            
            #print(f"Found Docker image: {container_name}")
        except Exception as e:
            print(f"Error checking Docker: {e}")
            print("Docker might not be running or not installed properly.")
            raise
    
    def solve_from_matrices(self, distance_matrix, precedence_matrix, cost_matrix, instance_name="custom_instance"):
        """Solve SOP using your matrices from CustomEnv."""
    
        # Create temp directory for file exchange
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create SOP file
            sop_file = os.path.join(temp_dir, f"{instance_name}.sop")
            self._create_sop_file(distance_matrix, precedence_matrix,cost_matrix, sop_file)
            
            # Print the actual file we created
            #with open(sop_file, 'r') as f:
                #print("Generated SOP file:")
                #print(f.read())
        
            # Try with different candidate list sizes
            for cand_size in [15, 10, 5]:
                try:
                    self._cand_list_size = cand_size
                    result = self.solve_from_file(sop_file)
                    if result:
                        return result
                    #print(f"Failed with candidate list size {cand_size}, trying smaller size...")
                except Exception as e:
                    print(f"Error with candidate list size {cand_size}: {e}")
                
        return None
    
    def solve_from_file(self, sop_file_path):
        """Solve SOP using an existing SOP file."""
        # Get absolute path
        abs_path = os.path.abspath(sop_file_path)
        file_name = os.path.basename(abs_path)
        dir_name = os.path.dirname(abs_path)
        
        #print(f"Using SOP file: {file_name}")
        #print(f"From directory: {dir_name}")
        
        # Create results directory in the host filesystem
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        #print(f"Results will be saved to: {results_dir}")
        
        # Extract dimension from file to set candidate list size
        dimension = self._get_dimension_from_file(sop_file_path)
        candidate_list_size = min(20, dimension - 1)  # Use a smaller candidate list
        #print(f"Using candidate list size: {candidate_list_size}")
        
        # Run the docker command
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        container_name = f"acssa_run_{timestamp}"
        
        cmd = [
            "docker", "run", "--name", container_name, "-d",
            "-v", f"{dir_name}:/data:ro",
            "-v", f"{results_dir}:/app/results:rw",
            self.container_name,
            "/bin/bash", "-c", 
            f"mkdir -p /app/results && chmod 777 /app/results && "
            f"./acs --alg=acs --ants=10 --phmem=std --ls=none --beta=2.0 "
            f"--test=/data/{file_name} --q0=10 --phi=0.01 --rho=0.1 "
            f"--cand_list={candidate_list_size} "
            f"--timeout={self.timeout} --trials=1"
        ]
        
        
        try:
            # Start the container
            container_id = subprocess.check_output(cmd).decode('utf-8').strip()
            #print(f"Started solver in container {container_id[:12]}")
            
            # Wait for container to finish
            while True:
                status = subprocess.check_output(
                    ["docker", "inspect", "--format={{.State.Status}}", container_id]
                ).decode('utf-8').strip()
                
                if status == "exited":
                    # Get the exit code
                    exit_code = subprocess.check_output(
                        ["docker", "inspect", "--format={{.State.ExitCode}}", container_id]
                    ).decode('utf-8').strip()
                    #print(f"Container exited with code {exit_code}")
                    
                    # Get logs
                    logs = subprocess.check_output(
                        ["docker", "logs", container_id]
                    ).decode('utf-8')
                    break
                    
                time.sleep(1)
                
        except Exception as e:
            print(f"Error running solver: {e}")
            return None
        
        # Get the results file
        result_directories = []
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.js'):
                    result_directories.append(os.path.join(root, file))

        #print(f"Found {len(result_directories)} .js files in results directory and subdirectories")

        if not result_directories:
            print("No result files found")
            return None
            
        latest_file = max(result_directories, key=os.path.getctime)

        # Parse the results
        try:
            with open(latest_file, 'r') as f:
                results = json.load(f)
            
            # Clean up the container
            subprocess.run(["docker", "rm", container_id], 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return results
            
        except Exception as e:
            print(f"Error parsing results: {e}")
            return None
    
    def _create_sop_file(self, distance_matrix, precedence_matrix, cost_matrix,output_file):
        """Convert the matrices to a SOP file format with special rows/columns."""
        n = distance_matrix.shape[0]
    
        
        distance_matrix = np.round(distance_matrix * 1000).astype(int)
        cost_matrix = np.round(cost_matrix * 1000).astype(int)
       
        with open(output_file, 'w') as f:
            # Write header - match format exactly from the example
            f.write(f"NAME: {os.path.basename(output_file)}\n")
            f.write("TYPE: SOP\n")
            f.write("COMMENT: Generated from Python CustomEnv matrices\n")
            f.write(f"DIMENSION: {n+2}\n")  # Dimension is now n+2
            f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX \n")  # Note the space after FULL_MATRIX
            f.write("EDGE_WEIGHT_SECTION\n")
            
            # Write the dimension
            f.write(f"{n+2} \n")
            
            # First row: [0, cost_values, 1000000]
            first_row = "  0 "  # Start with 0
            for i in range(n):
                first_row += f"{int(cost_matrix[i])} "
            first_row += "10000000 "
            f.write(first_row.rstrip() + " \n")
            
            # Middle rows: [-1, distance_matrix, 0]
            for i in range(n):
                line = "  -1 "  # First column is -1
                for j in range(n):
                    if i == j:
                        line += "0 "  # Diagonal is 0
                    elif distance_matrix[i, j] == 0 or precedence_matrix[j, i] == 1:
                        # If j must precede i (j->i), mark as impossible
                        line += "-1 "
                    else:
                        line += f"{distance_matrix[i, j]} "
                line += "0 "  # Last column is 0
                f.write(line.rstrip() + " \n")
            
            # Last row: [-1, -1, -1, ..., 0]
            last_row = "  " + " ".join(["-1"] * (n+1) + ["0"])
            f.write(last_row + " \n")
            
            f.write("EOF\n")
 
    def _get_dimension_from_file(self, file_path):
        """Extract the dimension from the SOP file."""
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("DIMENSION"):
                    dimension = int(line.split(":")[1].strip())
                    return dimension
        # Default if not found
        return 25

def main():
    
    # Initialize the solver
    solver = SOPSolver(container_name="acssa-builder", timeout=30)
    
    env = CustomEnv(25, "Random", normalize_rewards=True, p=0.01)
    # Get matrices from the environment
    distance_matrix = env.distance_matrix
    precedence_matrix = env.precedence_matrix
    cost_matrix = env.original_cost_matrix if hasattr(env, 'original_cost_matrix') else None

    # Unnormalize distance matrix if needed (ACS expects integer distances)
    if hasattr(env, 'max_value'):
        distance_matrix = distance_matrix * env.max_value

    # Solve using ACS
    #print("Solving with Ant Colony System...")
    solution = solver.solve_from_matrices(distance_matrix, precedence_matrix, cost_matrix)
    
    # If solution parsing fails, try to read the file directly
    if solution is None:
        results_dir = os.path.join(os.getcwd(), "results")
        json_files = glob.glob(os.path.join(results_dir, "**", "*.js"), recursive=True)
        
        if json_files:
            latest_file = max(json_files, key=os.path.getctime)
            print(f"Found results file: {latest_file}")
            
            # Read and print the raw contents
            with open(latest_file, 'r') as f:
                content = f.read()
                print("Raw JSON content:")
                print(content)
            
            # Try parsing it
            import json
            try:
                solution = json.loads(content)
            except json.JSONDecodeError:
                print("Failed to parse JSON")
    
    # Process and display the solution
    if solution:
        #print("\nSolution found!")
        
        # Check which keys are available in the solution
        #print("Available keys in solution:", solution.keys())
        
        # Use the correct key names based on what's available
        cost_key = 'best_solution_cost'
        path_key = 'best_solution'
        time_key = 'execution_time'
        
        # Try alternative key names if the expected ones don't exist
        if 'best_solution_cost' not in solution:
            if 'cost' in solution:
                cost_key = 'cost'
            elif 'best_cost' in solution:
                cost_key = 'best_cost'
                
        if 'best_solution' not in solution:
            if 'solution' in solution:
                path_key = 'solution'
            elif 'best_path' in solution:
                path_key = 'best_path'
                
        if 'execution_time' not in solution:
            if 'time' in solution:
                time_key = 'time'
            elif 'exec_time' in solution:
                time_key = 'exec_time'
        
        # Print the information using the correct keys
        print(f"Cost: {solution[cost_key]}")
        print(f"Path: {solution[path_key]}")
        print(f"Execution time: {solution[time_key] if time_key in solution else 'N/A'}")
        
        # Convert 1-indexed solution back to 0-indexed for visualization
        path = [x-1 for x in solution[path_key]]
        
        # Set the environment state to match the solution
        env.reset()
        for node in path:
            env.visited_nodes[node] = True
            
        env.actions_taken = path
        env.current_node = path[-1] if path else None
        
        # Visualize the solution
        env.render()

if __name__ == "__main__":
    main()