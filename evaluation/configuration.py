import argparse, math, os, subprocess, shutil

# Create an argument parser
parser = argparse.ArgumentParser(description="Run evaluation for the external face list algorithm")

# Add the 'method' argument
parser.add_argument('--method', type=int, default=0,
                    help="Evaluation method: 0 - all, 1 - memory footprint, 2 - CPU time, 3 - hash performance, 4 - parallel efficiency, 5 - GPU time."
                         "Default: 0")
parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for each evaluation. Default: 10")

# Parse the command-line arguments
args = parser.parse_args()

# Access the 'method' argument
method = args.method

# Iterations
iterations = args.iterations

# Directories
evaluation_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(evaluation_dir)
build_dir = os.path.join(src_dir, "build")  # Make sure this is the correct build directory
home_dir = os.path.expanduser("~")
datasets_dir = os.path.join(home_dir, "Data")  # Make sure this is the correct dataset directory

results_dir = os.path.join(evaluation_dir, "results")
# Make sure this is the correct configuration
# configuration = "testing"
# configuration = f"frontier_rocm5.7.0_tbb2021.13.0_kokkos4.1.00"
# configuration = f"frontier_rocm6.2.0_tbb2022.0.0_kokkos4.4.01"
configuration = f"frontier_rocm6.2.4_tbb2022.0.0_kokkos4.5.00"
config_dir = os.path.join(results_dir, configuration)

data_dir = os.path.join(config_dir, "data")
data_memory_footprint_dir = os.path.join(data_dir, "memory_footprint")
data_cpu_time_dir = os.path.join(data_dir, "cpu_time")
data_hash_performance_dir = os.path.join(data_dir, "hash_performance")
data_speed_up_dir = os.path.join(data_dir, "speed_up")
data_gpu_time_dir = os.path.join(data_dir, "gpu_time")

figures_dir = os.path.join(config_dir, "figures")
fig_memory_footprint_dir = os.path.join(figures_dir, "memory_footprint")
fig_cpu_time_dir = os.path.join(figures_dir, "cpu_time")
fig_hash_performance_dir = os.path.join(figures_dir, "hash_performance")
fig_speed_up_dir = os.path.join(figures_dir, "speed_up")
fig_gpu_time_dir = os.path.join(figures_dir, "gpu_time")

# Executables
executable = os.path.join(build_dir, "vtk-external-facelist-evaluation")
time_executable = shutil.which("time")
memory_evaluator = f"{time_executable} -v"
perf_executable = shutil.which("perf")
cache_evaluator = f"{perf_executable} stat -e cache-misses"

# Datasets # Make sure these are the correct datasets ordered from smallest to biggest
datasets = [f"{datasets_dir}/JSM.vtu",
            f"{datasets_dir}/F-15.vtu",
            f"{datasets_dir}/JSM-tet.vtu",
            f"{datasets_dir}/F-15-tet.vtu"]
biggest_datasets = datasets[2:]

# Algorithms
algorithms_names = {"--s-classifier": "S-Classifier", "--s-hash": "S-Hash", "--p-classifier": "P-Classifier",
                    "--p-hash": "P-Hash", "--p-hash-sort": "P-Hash-Sort", "--p-hash-fight": "P-Hash-Fight",
                    "--p-hash-count": "P-Hash-Count"}
algorithms = ["--s-classifier", "--s-hash", "--p-classifier", "--p-hash", "--p-hash-sort", "--p-hash-fight",
              "--p-hash-count"]
parallel_algorithms = algorithms[2:]
vtk_algorithms = algorithms[:4]
vtkm_algorithms = algorithms[4:]
algorithms_joined = " ".join(algorithms)
parallel_algorithms_joined = " ".join(parallel_algorithms)
vtkm_algorithms_joined = " ".join(vtkm_algorithms)

# Hash functions
hash_function_names = ["All", "FNV1A", "MinPointID"]
hash_functions = [0, 1, 2]

# 1 randomly generated seed from 0 to 2^32 - 1
random_seed = 2639962142

# Number of threads
# max_number_of_threads_power_of_2 = int(math.log2(os.cpu_count()))  # Make sure this uses the correct number of threads
max_number_of_threads_power_of_2 = int(math.log2(128))  # Make sure this uses the correct number of threads
max_number_of_threads = int(2 ** max_number_of_threads_power_of_2)


def get_dataset_name(filename):
    """Get the name of the dataset from the filename"""
    return os.path.splitext(os.path.basename(filename))[0]
    # return os.path.basename(filename)


# Function to run subprocess commands
def run_command(command, output_file):
    """Run a command and save the output to a file"""
    with open(output_file, 'a') as f:
        subprocess.run(command, shell=True, stdout=f, stderr=f)
