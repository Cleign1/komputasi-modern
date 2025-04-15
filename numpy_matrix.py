import numpy as np
import time
import os
import json
import psutil

# Define dimensions with additional large sizes to try
dimensi = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
# Very large dimensions that might fail - will only try these if others succeed
very_large_dimensi = [32768, 65536]

# Create directory to save results if it doesn't exist
os.makedirs("matrix_results", exist_ok=True)

# Dictionary to store execution times
numpy_times_dict = {}

# Get system memory information
mem_info = psutil.virtual_memory()
print(f"System memory information:")
print(f"Total RAM: {mem_info.total / (1024**3):.2f} GB")
print(f"Available RAM: {mem_info.available / (1024**3):.2f} GB")
print(f"Percent used: {mem_info.percent}%")

print("\nStarting NumPy matrix generation...")

# Function to check if we have enough memory for the array
def check_memory_available(dim):
    # Calculate required memory in bytes (8 bytes per int64)
    required_bytes = dim * dim * 8
    
    # Get available memory
    available_bytes = psutil.virtual_memory().available
    
    # Use only 80% of available memory as safety margin
    safe_available = available_bytes * 0.95
    
    sufficient_memory = required_bytes <= safe_available
    
    print(f"Matrix size: {dim}x{dim} requires {required_bytes / (1024**3):.2f} GB")
    print(f"Available memory: {available_bytes / (1024**3):.2f} GB (using 95%: {safe_available / (1024**3):.2f} GB)")
    print(f"Memory sufficient: {sufficient_memory}")
    
    return sufficient_memory

# Function to generate numpy matrix and measure time
def generate_numpy_matrix(dim):
    # Check if we have enough memory first
    if not check_memory_available(dim):
        print(f"Not enough memory for {dim}x{dim} matrix. Skipping.")
        return None, 0
    
    try:
        start_time = time.time()
        matrix = np.random.randint(1, 101, size=(dim, dim))
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"NumPy {dim}x{dim} matrix generated in {execution_time:.6f} seconds")
        return matrix, execution_time
    except (MemoryError, np._core._exceptions._ArrayMemoryError) as e:
        print(f"Memory error while allocating {dim}x{dim} matrix: {str(e)}")
        return None, 0
    except Exception as e:
        print(f"Unexpected error while allocating {dim}x{dim} matrix: {str(e)}")
        return None, 0

# Track if any failures occur to stop trying larger sizes
had_memory_failure = False

# Generate NumPy matrices for standard dimensions
for dim in dimensi:
    print(f"\nGenerating {dim}x{dim} NumPy matrix...")
    
    # Generate NumPy matrix
    np_matrix, np_time = generate_numpy_matrix(dim)
    
    if np_matrix is not None:
        numpy_times_dict[dim] = np_time
        
        # Save a sample of the NumPy matrix
        with open(f"matrix_results/numpy_matrix_{dim}x{dim}_sample.txt", "w") as f:
            f.write(f"NumPy Random Matrix {dim}x{dim} (showing first 10x10 elements):\n")
            sample_size = min(10, dim)
            for i in range(sample_size):
                f.write(" ".join(map(str, np_matrix[i, :sample_size])) + "\n")
        
        # Free memory explicitly
        del np_matrix
        import gc
        gc.collect()
        
        # Report memory after each large matrix
        if dim >= 1024:
            mem_info = psutil.virtual_memory()
            print(f"Available RAM after cleanup: {mem_info.available / (1024**3):.2f} GB")
    else:
        had_memory_failure = True
        break

# Try very large dimensions only if all previous ones succeeded
if not had_memory_failure:
    for dim in very_large_dimensi:
        print(f"\nAttempting to generate very large {dim}x{dim} NumPy matrix...")
        
        # Generate NumPy matrix
        np_matrix, np_time = generate_numpy_matrix(dim)
        
        if np_matrix is not None:
            numpy_times_dict[dim] = np_time
            
            # Save a sample of the NumPy matrix
            with open(f"matrix_results/numpy_matrix_{dim}x{dim}_sample.txt", "w") as f:
                f.write(f"NumPy Random Matrix {dim}x{dim} (showing first 10x10 elements):\n")
                sample_size = min(10, dim)
                for i in range(sample_size):
                    f.write(" ".join(map(str, np_matrix[i, :sample_size])) + "\n")
            
            # Free memory explicitly
            del np_matrix
            import gc
            gc.collect()
            
            # Report memory after each matrix
            mem_info = psutil.virtual_memory()
            print(f"Available RAM after cleanup: {mem_info.available / (1024**3):.2f} GB")
        else:
            print(f"Skipping remaining very large dimensions")
            break

# Save timing results to a JSON file
with open("matrix_results/numpy_times.json", "w") as f:
    json.dump(numpy_times_dict, f)

print("\nNumPy matrix generation completed.")
print(f"Results saved to matrix_results/numpy_times.json")

# Print summary
print("\nNumPy Summary:")
print("=" * 50)
print(f"{'Dimension':<10} {'Time (seconds)':<15}")
print("-" * 50)
for dim in sorted(numpy_times_dict.keys()):
    print(f"{dim:<10} {numpy_times_dict[dim]:<15.6f}")