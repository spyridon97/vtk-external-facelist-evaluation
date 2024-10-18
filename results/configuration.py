import argparse, os, math, subprocess, shutil

# Create an argument parser
parser = argparse.ArgumentParser(description="Run evaluation for the external face list algorithm")

# Add the 'method' argument
parser.add_argument('--method', type=int, default=0,
                    help="Evaluation method: 0 - all, 1 - memory footprint, 2 - CPU time, 3 - hash distribution, 4 - parallel efficiency, 5 - GPU time."
                         "Default: 0")
parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for each evaluation. Default: 10")

# Parse the command-line arguments
args = parser.parse_args()

# Access the 'method' argument
method = args.method

# Iterations
iterations = args.iterations

# Directories
results_dir = os.path.dirname(os.path.abspath(__file__))
memory_footprint_dir = os.path.join(results_dir, "memory_footprint")
cpu_time_dir = os.path.join(results_dir, "cpu_time")
hash_performance_dir = os.path.join(results_dir, "hash_performance")
parallel_efficiency_dir = os.path.join(results_dir, "cpu_parallel_efficiency")
gpu_time_dir = os.path.join(results_dir, "gpu_time")

src_dir = os.path.dirname(results_dir)
# build_dir = os.path.join(src_dir, "cmake-build-relwithdebinfo")  # Make sure this is the correct build directory
build_dir = os.path.join(src_dir, "build")  # Make sure this is the correct build directory
home_dir = os.path.expanduser("~")
# datasets_dir = os.path.join(home_dir, "Programming/Data")  # Make sure this is the correct dataset directory
datasets_dir = os.path.join(home_dir, "Data")  # Make sure this is the correct dataset directory

# Executables
executable = os.path.join(build_dir, "vtk-external-facelist-evaluation")
time_executable = shutil.which("time")
memory_evaluator = f"{time_executable} -v"
perf_executable = shutil.which("perf")
cache_evaluator = f"{perf_executable} stat -e cache-misses"

# Datasets # Make sure these are the correct datasets ordered from smallest to biggest
# datasets = [f"{datasets_dir}/JSM_Case1_UnstrMixed_JAXA_mediumRev00.vtu",
#             f"{datasets_dir}/JSM_Case1_UnstrMixed_JAXA_mediumRev00-tetra.vtu",
#             f"{datasets_dir}/Helden/F15_r2.0/F-15.vtu",
#             f"{datasets_dir}/Wavelet300Tetra.vtu"]
datasets = [f"{datasets_dir}/JSM.vtu",
            f"{datasets_dir}/JSM-tet.vtu",
            f"{datasets_dir}/F-15.vtu",
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
hash_function_names = ["Both", "FNV1A", "MinPointID"]
hash_functions = [0, 1, 2]

# Number of threads
max_number_of_threads_power_of_2 = int(math.log2(os.cpu_count()))
max_number_of_threads = int(2 ** max_number_of_threads_power_of_2)


def get_dataset_name(filename):
    return os.path.splitext(os.path.basename(filename))[0]
    # return os.path.basename(filename)
    # TODO Use the above line for old data


# Function to run subprocess commands
def run_command(command, output_file):
    with open(output_file, 'a') as f:
        # subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
        subprocess.run(command, shell=True, stdout=f, stderr=f)
