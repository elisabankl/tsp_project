import os
import numpy as np

def create_sop_file(distance_matrix, precedence_matrix, cost_matrix, output_file):
    """Convert matrices to SOP file format with special rows/columns.
    
    This creates an (n+2)x(n+2) matrix with:
    - First row: [0, cost_values, 1000000]
    - First column: [0, -1, -1, ..., -1]
    - Last column: [1000000, 0, 0, ..., 0]
    - Last row: [-1, -1, ..., -1, 0]
    
    Args:
        distance_matrix: NxN matrix of distances between nodes
        precedence_matrix: NxN matrix where (i,j)=1 means i must precede j
        cost_matrix: N-vector of node costs
        output_file: Path to save the SOP file
    """
    n = distance_matrix.shape[0]
    
    # Convert matrices to integers
    if np.max(distance_matrix) <= 1.0:
        distance_matrix = np.round(distance_matrix * 100).astype(int)
        cost_matrix = np.round(cost_matrix * 100).astype(int)
    else:
        distance_matrix = np.round(distance_matrix).astype(int)
        cost_matrix = np.round(cost_matrix).astype(int)
    
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
        first_row += "1000000 "
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
    
    print(f"Created SOP file: {output_file}")
    # Print the first few lines of the created file
    with open(output_file, 'r') as f:
        first_lines = "".join([f.readline() for _ in range(min(15, n+15))])
        print(f"First lines of the file:\n{first_lines}...")
        
    return output_file