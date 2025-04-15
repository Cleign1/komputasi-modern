import matplotlib.pyplot as plt
import json
import os

# Create directory for plots if it doesn't exist
os.makedirs("matrix_results", exist_ok=True)

# Load timing data
try:
    with open("matrix_results/numpy_times.json", "r") as f:
        numpy_times_dict = json.load(f)
    print("Loaded NumPy timing data")
except FileNotFoundError:
    print("NumPy timing data not found")
    numpy_times_dict = {}

try:
    with open("matrix_results/cupy_times.json", "r") as f:
        cupy_times_dict = json.load(f)
    print("Loaded CuPy timing data")
except FileNotFoundError:
    print("CuPy timing data not found")
    cupy_times_dict = {}

# Convert string keys to integers (JSON converts keys to strings)
numpy_times_dict = {int(k): v for k, v in numpy_times_dict.items()}
cupy_times_dict = {int(k): v for k, v in cupy_times_dict.items()}

if numpy_times_dict and cupy_times_dict:
    # Create plot
    plt.figure(figsize=(12, 7))
    
    # Plot NumPy times
    np_dims = sorted(numpy_times_dict.keys())
    np_times = [numpy_times_dict[d] for d in np_dims]
    plt.plot(np_dims, np_times, 'o-', label='NumPy', color='blue')
    
    # Plot CuPy times
    cp_dims = sorted(cupy_times_dict.keys())
    cp_times = [cupy_times_dict[d] for d in cp_dims]
    plt.plot(cp_dims, cp_times, 's-', label='CuPy', color='green')
    
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison: NumPy vs CuPy for Random Matrix Generation')
    plt.legend()
    
    # Use 'log' scale
    plt.xscale('log')
    plt.yscale('log')
    
    # Add grid for better readability
    plt.grid(True, which="both", ls="-")
    
    # Add custom x-ticks for dimensions that were successful
    all_successful_dims = sorted(list(set(list(numpy_times_dict.keys()) + list(cupy_times_dict.keys()))))
    plt.xticks(all_successful_dims, [str(d) for d in all_successful_dims], rotation=45)
    
    plt.tight_layout()
    plt.savefig('matrix_results/performance_comparison.png')
    print("Performance comparison plot saved to 'matrix_results/performance_comparison.png'")
    
    # Calculate and print speedup
    print("\nSpeedup Summary:")
    print("=" * 70)
    print(f"{'Dimension':<10} {'NumPy Time (s)':<15} {'CuPy Time (s)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    common_dims = sorted(set(numpy_times_dict.keys()) & set(cupy_times_dict.keys()))
    
    for dim in common_dims:
        np_time = numpy_times_dict.get(dim, float('nan'))
        cp_time = cupy_times_dict.get(dim, float('nan'))
        
        if np_time > 0 and cp_time > 0:
            speedup = np_time / cp_time
            print(f"{dim:<10} {np_time:<15.6f} {cp_time:<15.6f} {speedup:<10.2f}x")
        else:
            print(f"{dim:<10} {np_time:<15.6f} {cp_time:<15.6f} {'N/A':<10}")
    
    plt.show()
else:
    print("Not enough data to create a comparison plot. Make sure both NumPy and CuPy scripts have been run.")