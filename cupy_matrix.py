import cupy as cp
import numpy as np
import time
import os
import json
import subprocess

# Define dimensions - start with smaller sizes and add larger ones
dimensi_gpu = [8, 16, 32, 64, 128, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
# Larger dims to try if memory allows
# larger_dims = [2048, 4096]

# Create directory to save results if it doesn't exist
os.makedirs("matrix_results", exist_ok=True)

# Dictionary to store execution times
cupy_times_dict = {}

print("Starting CuPy matrix generation...")

# Run nvidia-smi to show GPU information
print("\n--- NVIDIA GPU Information ---")
try:
    nvidia_smi_output = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
    print(nvidia_smi_output)
except (subprocess.SubprocessError, FileNotFoundError) as e:
    print(f"Error running nvidia-smi: {e}")

# Also get CuPy's memory info for precise calculations
total_gpu_memory = cp.cuda.runtime.memGetInfo()[1]
free_gpu_memory = cp.cuda.runtime.memGetInfo()[0]
print(f"Total GPU memory (reported by CuPy): {total_gpu_memory / (1024**3):.2f} GB")
print(f"Free GPU memory (reported by CuPy): {free_gpu_memory / (1024**3):.2f} GB")
print("-------------------------------\n")

# Function to generate cupy matrix with memory safety checks
def generate_cupy_matrix(dim):
    # Calculate required memory and check if enough is available
    required_bytes = dim * dim * 8  # 8 bytes per float64
    available_memory = cp.cuda.runtime.memGetInfo()[0]  # Get free memory
    
    if required_bytes > available_memory * 0.95:  # Use only 80% of available memory as safety
        print(f"Warning: Not enough GPU memory for {dim}x{dim} matrix.")
        print(f"Required: {required_bytes / (1024**3):.2f} GB, Available: {available_memory / (1024**3):.2f} GB")
        return None, 0
    
    try:
        start_time = time.time()
        matrix = cp.random.randint(1, 101, size=(dim, dim))
        # Force synchronization to get accurate timing
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"CuPy {dim}x{dim} matrix generated in {execution_time:.6f} seconds")
        return matrix, execution_time
    except cp.cuda.memory.OutOfMemoryError:
        print(f"Out of memory error for {dim}x{dim} matrix.")
        return None, 0

# Generate CuPy matrices for standard dimensions
for dim in dimensi_gpu:
    print(f"\nGenerating {dim}x{dim} CuPy matrix...")
    
    # Generate CuPy matrix
    cp_matrix, cp_time = generate_cupy_matrix(dim)
    
    if cp_matrix is not None:
        cupy_times_dict[dim] = cp_time
        
        # Save a sample of the CuPy matrix
        with open(f"matrix_results/cupy_matrix_{dim}x{dim}_sample.txt", "w") as f:
            f.write(f"CuPy Random Matrix {dim}x{dim} (showing first 10x10 elements):\n")
            sample_size = min(10, dim)
            cp_matrix_np = cp.asnumpy(cp_matrix)  # Convert to NumPy array for saving
            for i in range(sample_size):
                f.write(" ".join(map(str, cp_matrix_np[i, :sample_size])) + "\n")
        
        # Free memory explicitly
        del cp_matrix
        cp.get_default_memory_pool().free_all_blocks()
        print(f"Free GPU memory after cleanup: {cp.cuda.runtime.memGetInfo()[0] / (1024**3):.2f} GB")
        
        # Run nvidia-smi after each large matrix to monitor GPU state

# # Try larger dimensions if previous ones succeeded
# for dim in larger_dims:
#     print(f"\nAttempting to generate larger {dim}x{dim} CuPy matrix...")
    
#     # Generate CuPy matrix
#     cp_matrix, cp_time = generate_cupy_matrix(dim)
    
#     if cp_matrix is not None:
#         cupy_times_dict[dim] = cp_time
        
#         # Save a sample of the CuPy matrix
#         with open(f"matrix_results/cupy_matrix_{dim}x{dim}_sample.txt", "w") as f:
#             f.write(f"CuPy Random Matrix {dim}x{dim} (showing first 10x10 elements):\n")
#             sample_size = min(10, dim)
#             cp_matrix_np = cp.asnumpy(cp_matrix)  # Convert to NumPy array for saving
#             for i in range(sample_size):
#                 f.write(" ".join(map(str, cp_matrix_np[i, :sample_size])) + "\n")
        
#         # Free memory explicitly
#         del cp_matrix
#         cp.get_default_memory_pool().free_all_blocks()
#         print(f"Free GPU memory after cleanup: {cp.cuda.runtime.memGetInfo()[0] / (1024**3):.2f} GB")
        
#         # Run nvidia-smi after each large matrix
#         try:
#             nvidia_smi_output = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
#             print("\n--- GPU state after matrix generation ---")
#             print(nvidia_smi_output)
#             print("-------------------------------\n")
#         except (subprocess.SubprocessError, FileNotFoundError):
#             pass
#     else:
#         print(f"Skipping remaining larger dimensions")
#         break

# Save timing results to a JSON file
with open("matrix_results/cupy_times.json", "w") as f:
    json.dump(cupy_times_dict, f)

print("\nCuPy matrix generation completed.")
print(f"Results saved to matrix_results/cupy_times.json")

# Print summary
print("\nCuPy Summary:")
print("=" * 50)
print(f"{'Dimension':<10} {'Time (seconds)':<15}")
print("-" * 50)
for dim in sorted(cupy_times_dict.keys()):
    print(f"{dim:<10} {cupy_times_dict[dim]:<15.6f}")