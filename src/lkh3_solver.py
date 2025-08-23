# filename: lkh_solver.py
import os
import json
import subprocess
import numpy as np
import tempfile
from datetime import datetime
import time
import glob

# Import the shared function
from tsp_utils import create_sop_file

class LKHSolver:
    def __init__(self, container_name="305c3345be25", timeout=10): #"vsr-lkh3"
        """Initialize the LKH solver with container configuration."""
        self.container_name = container_name
        self.timeout = timeout
        
        # Verify container exists
        try:
            result = subprocess.run(["docker", "image", "inspect", container_name], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result)
            if result.returncode != 0:
                print(f"Container {container_name} not found. Please build it first.")
                print("Run: docker build -t vsr-lkh3 -f Dockerfile-LKH3 .")
                raise RuntimeError(f"Container {container_name} not found")
            
            print(f"Found Docker image: {container_name}")
        except Exception as e:
            print(f"Error checking Docker: {e}")
            raise
    
    def solve_from_matrices(self, distance_matrix, precedence_matrix, cost_matrix=None, instance_name="custom_instance"):
        """Solve SOP/TSP using matrices."""
        # Create temp directory for file exchange
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create problem file
            sop_file = os.path.join(temp_dir, f"{instance_name}.sop")
            par_file = os.path.join(temp_dir, f"{instance_name}.par")
            tour_file = os.path.join(temp_dir, f"{instance_name}.tour")
            
            # Make sure cost_matrix exists (even if all zeros)
            if cost_matrix is None:
                cost_matrix = np.zeros(distance_matrix.shape[0])
            
            # Use the shared utility function
            create_sop_file(distance_matrix, precedence_matrix, cost_matrix, sop_file)
            
            # Print the actual file we created
            with open(sop_file, 'r') as f:
                print("Generated SOP file:")
                print(f.read())
            
            # Create parameter file
            self._create_par_file(par_file, instance_name)
            
            # Run the solver
            result = self.solve_from_files(temp_dir, par_file, tour_file)
            return result
    
    def solve_from_files(self, work_dir, par_file, tour_file):
        """Run VSR-LKH-3 on existing files."""
        # Get absolute paths
        abs_work_dir = os.path.abspath(work_dir)
        par_filename = os.path.basename(par_file)
        tour_filename = os.path.basename(tour_file)
        
        # Run the docker command
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        container_name = f"lkh_run_{timestamp}"
        
        # Improved debugging for Docker command
        # Print files in the temp directory before running Docker
        print(f"Files in {abs_work_dir} before Docker run:")
        try:
            files = os.listdir(abs_work_dir)
            for file in files:
                print(f"  - {file}")
        except Exception as e:
            print(f"Error listing files: {e}")
    
        # Modified Docker run command with improved reliability
        cmd = [
            "docker", "run", "--name", container_name, "-d",
            "-v", f"{abs_work_dir}:/data",
            self.container_name,
            "/bin/bash", "-c", 
            f"cd /app/vsr-lkh-v2/VSR-LKH-3-Final && "
            f"ls -la /data && "  # List files in mounted directory
            f"cp /data/*.sop . && "  # Copy SOP file first
            f"cp /data/{par_filename} . && "
            f"ls -la && "  # List files in working directory
            f"./LKH {par_filename} && "
            f"ls -la && "  # List files after LKH runs
            f'find / -name "*.tour" -type f && '
            f"cd TOURS &&"
            f"ls -la && "  # List files in TOURS directory
            f"cp TOURS/*.tour /data/ || echo 'No tour files to copy'"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            # Start the container
            container_id = subprocess.check_output(cmd).decode('utf-8').strip()
            print(f"Started solver in container {container_id[:12]}")
            
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
                    print(f"Container exited with code {exit_code}")
                    
                    # Get logs
                    logs = subprocess.check_output(
                        ["docker", "logs", container_id]
                    ).decode('utf-8')
                    print("Container logs:")
                    print(logs)
                    break
                    
                print("Solving...", end="\r")
                time.sleep(1)
                
            # Check if tour file was created
            tour_path = os.path.join(abs_work_dir, tour_filename)
            if os.path.exists(tour_path):
                result = self._parse_tour_file(tour_path)
                
                # Clean up the container
                subprocess.run(["docker", "rm", container_id], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                return result
            else:
                # Look for any tour files
                tour_files = glob.glob(os.path.join(abs_work_dir, "*.tour"))
                if tour_files:
                    print(f"Found tour files: {tour_files}")
                    return self._parse_tour_file(tour_files[0])
                else:
                    print("No tour file was created")
                    return None
                
        except Exception as e:
            print(f"Error running solver: {e}")
            return None
    
    def _create_par_file(self, par_file_path, instance_name):
        """Create a parameter file for LKH-3."""
        sop_filename = f"{instance_name}.sop"  # Use local path within container working dir
        tour_filename = f"{instance_name}.tour"
        
        with open(par_file_path, 'w') as f:
            f.write(f"PROBLEM_FILE = {sop_filename}\n")  # Local file in working dir
            f.write(f"OUTPUT_TOUR_FILE = {tour_filename}\n")
            f.write("RUNS = 5\n")  # Reduced runs for faster testing
            f.write("MAX_TRIALS = 5000\n")
            f.write("TRACE_LEVEL = 3\n")  # Increased trace level for debugging
            f.write("CANDIDATE_SET_TYPE = ALPHA\n")
            f.write("SEED = 42\n")
            f.write(f"TIME_LIMIT = {self.timeout}\n")
            f.write("EOF\n")
    
        print(f"Created parameter file: {par_file_path}")
    
    def _parse_tour_file(self, tour_file):
        """Parse the LKH tour output file."""
        try:
            with open(tour_file, 'r') as f:
                lines = f.readlines()
                
            # Look for tour section
            tour_section = False
            tour = []
            cost = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Extract cost from the header comments
                if "Cost" in line and cost is None:
                    parts = line.split('=')
                    if len(parts) > 1:
                        cost_str = parts[1].strip()
                        if cost_str.startswith('0_'):  # Format like "0_572"
                            cost = int(cost_str.split('_')[1])
                        else:
                            try:
                                cost = int(cost_str)
                            except ValueError:
                                pass
                
                # Extract the tour
                if line == "TOUR_SECTION":
                    tour_section = True
                    continue
                elif line == "-1" or line == "EOF":
                    tour_section = False
                    continue
                
                if tour_section and line:
                    try:
                        node = int(line)
                        tour.append(node)
                    except ValueError:
                        pass
            
            # If no cost found in header, try to calculate from successes/runs line
            if cost is None:
                for line in lines:
                    if "Cost.min =" in line:
                        parts = line.split('=')
                        if len(parts) > 1:
                            try:
                                cost = int(parts[1].split(',')[0].strip())
                            except ValueError:
                                pass
            
            # Format the result like the ACS solver for compatibility
            return {
                "runs": [{
                    "best_solution": tour,
                    "best_solution_value": cost
                }],
                "total_time_sec": 0.0,  # Could parse this from the file if needed
                "tour_file": tour_file
            }
            
        except Exception as e:
            print(f"Error parsing tour file: {e}")
            return None

    def test_local_files(self):
        """Test the file generation locally without Docker."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a small test instance
            n = 5
            distance_matrix = np.random.randint(1, 10, size=(n, n))
            np.fill_diagonal(distance_matrix, 0)  # Zero diagonal
            precedence_matrix = np.zeros((n, n))  # No precedence constraints
            cost_matrix = np.random.randint(1, 5, size=n)
            
            # Create problem files
            sop_file = os.path.join(temp_dir, "test.sop")
            par_file = os.path.join(temp_dir, "test.par")
            
            # Generate SOP file
            from tsp_utils import create_sop_file
            create_sop_file(distance_matrix, precedence_matrix, cost_matrix, sop_file)
            
            # Generate PAR file
            with open(par_file, 'w') as f:
                f.write("PROBLEM_FILE = test.sop\n")
                f.write("OUTPUT_TOUR_FILE = test.tour\n")
                f.write("RUNS = 1\n")
                f.write("MAX_TRIALS = 100\n")
                f.write("TRACE_LEVEL = 3\n")
                f.write("EOF\n")
            
            print(f"\nTest files created in {temp_dir}")
            print("You can manually test these files with LKH")
            
            # Print the generated SOP file for inspection
            with open(sop_file, 'r') as f:
                print("\nGenerated SOP file:")
                print(f.read())
            
            return temp_dir

def main():
    # Example usage
    from customenv import CustomEnv
    
    # Create an environment instance
    env = CustomEnv(25, "Euclidic", p=0.1)
    
    # Create the solver
    solver = LKHSolver()  
    
    # Get matrices from the environment
    distance_matrix = env.distance_matrix 
    precedence_matrix = env.precedence_matrix
    
    # Unnormalize distance matrix if needed
    if hasattr(env, 'max_value'):
        distance_matrix = distance_matrix * env.max_value
    
    # Solve using LKH
    print("Solving with VSR-LKH-3...")
    solution = solver.solve_from_matrices(distance_matrix, precedence_matrix)
    
    # Process and display the solution
    if solution:
        print("\nSolution found!")
        print(f"Best tour: {solution['runs'][0]['best_solution']}")
        print(f"Best tour cost: {solution['runs'][0]['best_solution_value']}")
        
        # Convert 1-indexed solution back to 0-indexed for visualization
        path = [x-1 for x in solution['runs'][0]['best_solution']]
        
        # Filter out any artificial nodes (those >= N)
        path = [node for node in path if node < env.N_GRAPH_SIZE]
        
        # Calculate the solution cost manually
        total_cost = 0
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            edge_cost = distance_matrix[from_node, to_node]
            total_cost += edge_cost
            print(f"Edge {from_node}->{to_node}: Cost = {edge_cost}")
        
        print(f"Manually calculated cost: {total_cost}")
        
        # Visual confirmation of the path
        print("Path:", " -> ".join(map(str, path)))
    else:
        print("Failed to find a solution")
    
if __name__ == "__main__":
    main()