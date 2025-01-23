import numpy as np
import re, yaml
import pandas as pd
import matplotlib.pyplot as plt
from configuration import *

# Generate figures using the results

# Figure size
fig_width = 8
fig_height = 6.5

legend_fontsize = 11
axis_label_fontsize = 14
title_fontsize = axis_label_fontsize


# Set column width and spacing x-axis positions for datasets
def get_width_and_spacing(num_algorithms):
    max_num_algorithms = len(vtk_algorithms) + (len(hash_functions) - 1) * len(vtkm_algorithms)
    ratio = max_num_algorithms / num_algorithms
    width = 0.10 * ratio
    spacing = 1.2
    return width, spacing


def load_gpl_palette(file_path):
    with open(file_path, 'r') as file:
        colors = []
        for line in file:
            # Skip lines that start with '#' or are empty
            if line.startswith('#') or not line.strip():
                continue

            # Split the line into components
            parts = line.split()
            if len(parts) >= 4:
                # Extract RGB values and name (the last part)
                r, g, b = map(float, parts[:3])  # Convert RGB to integers
                r /= 255
                g /= 255
                b /= 255
                # name = ' '.join(parts[3:])  # Join the remaining parts as the color name
                colors.append((r, g, b))  # Store as a tuple (R, G, B, Name)
        return colors


# Load palette
# palette = [plt.cm.tab10(i) for i in range(len(vtk_algorithms) + 2 * len(vtkm_algorithms))]
palette = load_gpl_palette(f'{evaluation_dir}/Paired_10.gpl')

# Create a dictionary to map algorithms to colors
algorithm_colors = {}
for i, algo in enumerate(vtk_algorithms):
    algorithm_name = algorithms_names[algo]
    algorithm_label = f"{algorithm_name}-MinPointID" if "Hash" in algorithm_name else algorithm_name
    algorithm_colors[algorithm_label] = palette[i]
for i, algo in enumerate(vtkm_algorithms):
    for hash_function in hash_functions[1:]:
        algorithm_name = f"{algorithms_names[algo]}-{hash_function_names[hash_function]}"
        algorithm_colors[algorithm_name] = palette[
            len(hash_functions[1:]) * i + (hash_function - 1) + len(vtk_algorithms)]
# reorder the colors so that S-Classifier is next to P-Classifier
algorithm_colors["P-Classifier"], algorithm_colors["S-Hash-MinPointID"] = algorithm_colors["S-Hash-MinPointID"], \
    algorithm_colors["P-Classifier"]


def print_improvement_ratio(df):
    print()
    # find me ratios of all algorithms over the minimum valud for each dataset
    for dataset in df.columns:
        min = df[dataset].min()
        for algo in df.index:
            value = df.loc[algo][dataset]
            ratio = value / min
            print(f"Dataset: {dataset}, Algorithm: {algo}, Value/Min Ratio: {ratio:.2f}")
        print()
    algorithm_ratios = {}
    for dataset in df.columns:
        min = df[dataset].min()
        for algorithm in df.index:
            value = df.loc[algorithm][dataset]
            ratio = value / min
            if algorithm not in algorithm_ratios:
                algorithm_ratios[algorithm] = []
            algorithm_ratios[algorithm].append(ratio)
    for algorithm, ratios in algorithm_ratios.items():
        print(f"Algorithm: {algorithm}, Min - Max Ratios: {np.min(ratios):.2f}x - {np.max(ratios):.2f}x")
    print()


def create_bar_chart(df, add_min_offset, x_label, y_label, legend_title, legend_loc, figure_filename):
    # Number of datasets and algorithms
    num_datasets = len(df.columns)
    num_algorithms = len(df.index)

    # Create the figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    width, spacing = get_width_and_spacing(num_algorithms)

    # Set the x-axis positions for datasets
    dataset_positions = np.arange(num_datasets) * spacing

    # Loop through each algorithm and plot its values across datasets
    for j, algorithm in enumerate(df.index):
        # Extract values for the algorithm
        values = df.loc[algorithm].values

        # Plot each algorithm's memory footprint as a horizontal bar for each dataset
        ax.barh(y=dataset_positions + ((num_algorithms - 1) - j - (num_algorithms - 1) / 2) * width, width=values,
                height=width, label=algorithm, color=algorithm_colors[algorithm])

    # Loop through each dataset to find and plot its minimum value
    for i, dataset in enumerate(df.columns):
        # Calculate the minimum value for the current dataset
        min_value = df[dataset].min()
        if add_min_offset:
            # add a small offset to the minimum value to avoid overlapping with the vertical line
            min_value *= 0.99

        # Plot a horizontal line at the correct position for each dataset's minimum value
        ax.vlines(x=min_value,
                  ymin=dataset_positions[i] - width * num_algorithms / 2,  # Start of line
                  ymax=dataset_positions[i] + width * num_algorithms / 2,  # End of line
                  color='grey', linestyle='--', linewidth=1)

    # Set the y-ticks to represent datasets
    ax.set_yticks(dataset_positions)
    ax.set_yticklabels(df.columns)

    # Add labels and title
    ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
    # ax.set_title('Title', fontsize=title_fontsize)

    # Add legend
    ax.legend(title=legend_title, loc=legend_loc, fontsize=legend_fontsize)

    # Show grid lines for clarity
    ax.grid(axis='x', linestyle='--')

    plt.tight_layout()
    plt.savefig(figure_filename, dpi=300, bbox_inches='tight')


if method == 0 or method == 1:
    print("Print Memory Footprint information")
    os.makedirs(fig_memory_footprint_dir, exist_ok=True)


    def get_memory_footprint_info(filename):
        # Read the file
        with open(filename, "r") as file:
            content = file.read()
            # Use regular expressions to find the values
            dataset_memory_used_match = re.search(r"dataset-memory-used:\s*(\d+)", content)
            max_resident_set_size_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", content)
            # Extract the values if found
            if dataset_memory_used_match and max_resident_set_size_match:
                dataset_memory_used = int(dataset_memory_used_match.group(1))
                max_resident_set_size = int(max_resident_set_size_match.group(1))
                algorithm_memory_used = (max_resident_set_size - dataset_memory_used) / (1024 * 1024)  # Convert to GB
                return algorithm_memory_used
            else:
                print("Error while reading the file")
                return 0


    memory_footprint_data = {}
    for dataset in datasets:
        dataset_name = get_dataset_name(dataset)
        memory_footprint_data[dataset_name] = {}

        for algo in vtk_algorithms:
            algorithm_name = algorithms_names[algo]
            algorithm_label = f"{algorithm_name}-MinPointID" if "Hash" in algorithm_name else algorithm_name
            output_file = f"{data_memory_footprint_dir}/{dataset_name}_{algorithm_name}.txt"
            memory_footprint_data[dataset_name][algorithm_label] = get_memory_footprint_info(output_file)

        for algo in vtkm_algorithms:
            for hash_function in hash_functions[1:]:
                algorithm_name = algorithms_names[algo] + "-" + hash_function_names[hash_function]
                output_file = f"{data_memory_footprint_dir}/{dataset_name}_{algorithm_name}.txt"
                memory_footprint_data[dataset_name][algorithm_name] = get_memory_footprint_info(output_file)

    # Convert the data to a pandas DataFrame
    df_memory_footprint = pd.DataFrame(memory_footprint_data)
    df_memory_footprint.index.name = 'Algorithm'
    # swap S-Classifier is next to P-Classifier
    algorithms_list = df_memory_footprint.index.tolist()
    algorithms_list[1], algorithms_list[2] = algorithms_list[2], algorithms_list[1]
    df_memory_footprint = df_memory_footprint.reindex(algorithms_list)
    df_memory_footprint.to_csv(f"{fig_memory_footprint_dir}/memory_footprint.csv", index=True, header=True)
    print(df_memory_footprint)

    # Print improvement ratios
    print_improvement_ratio(df_memory_footprint)

    # Create bar chart
    create_bar_chart(df_memory_footprint, True, 'Memory footprint (gigabytes)', 'Datasets', 'Algorithms', 'lower right',
                     f"{fig_memory_footprint_dir}/memory_footprint.png")


def get_experiment_average_time_of_algorithm(experiment):
    algorithm_name = experiment['algorithm-name']
    hash_function = experiment['hash-name']
    algorithm_full_name = f"{algorithm_name}-{hash_function}" if hash_function != "None" else algorithm_name
    trials = experiment.get('trials', [])
    total_time = sum(trial['seconds-total'] for trial in trials)
    avg_time = total_time / len(trials) if trials else 0
    return algorithm_full_name, avg_time


def get_run_time_info(filename):
    try:
        # Read the yaml file
        with open(filename, "r") as file:
            content = yaml.load(file, Loader=yaml.FullLoader)[0]
            experiments = content["experiments"]
            # print(content)
            time_data_per_algorithm = {}
            for experiment in experiments:
                algorithm_full_name, avg_time = get_experiment_average_time_of_algorithm(experiment)
                time_data_per_algorithm[algorithm_full_name] = avg_time
                # print(f"Algorithm: {algorithm_full_name}, Average seconds-total: {avg_time:.4f}")
            return time_data_per_algorithm
    except (yaml.YAMLError, IndexError) as e:
        # print(f"Error parsing YAML file: {e}")
        return None  # Or handle the error differently, e.g., raise an exception


if method == 0 or method == 2:
    print("Print CPU Time information")
    os.makedirs(fig_cpu_time_dir, exist_ok=True)

    cpu_time_data_1_threads_normal = {}
    cpu_time_data_max_threads_normal = {}
    cpu_time_data_max_threads_random = {}
    for dataset in datasets:
        dataset_name = get_dataset_name(dataset)

        time_1_threads_normal_file = f"{data_cpu_time_dir}/{dataset_name}_1_threads_normal.yaml"
        cpu_time_data_1_threads_normal[dataset_name] = get_run_time_info(time_1_threads_normal_file)

        time_max_threads_normal_file = f"{data_cpu_time_dir}/{dataset_name}_{max_number_of_threads}_threads_normal.yaml"
        cpu_time_data_max_threads_normal[dataset_name] = get_run_time_info(time_max_threads_normal_file)

        time_max_threads_random_file = f"{data_cpu_time_dir}/{dataset_name}_{max_number_of_threads}_threads_random_{random_seed}.yaml"
        cpu_time_data_max_threads_random[dataset_name] = get_run_time_info(time_max_threads_random_file)

    filename_prefixes = [f"{fig_cpu_time_dir}/cpu_time_1_threads_normal",
                         f"{fig_cpu_time_dir}/cpu_time_{max_number_of_threads}_threads_normal",
                         f"{fig_cpu_time_dir}/cpu_time_{max_number_of_threads}_threads_random_{random_seed}"]
    df_cpu_time_data = [cpu_time_data_1_threads_normal, cpu_time_data_max_threads_normal,
                        cpu_time_data_max_threads_random]
    for filename_prefix, cpu_time_data in zip(filename_prefixes, df_cpu_time_data):
        print(f"Processing file: {filename_prefix}")
        # Convert the data to a pandas DataFrame
        df_cpu_time = pd.DataFrame(cpu_time_data)
        df_cpu_time.index.name = 'Algorithm'
        if filename_prefix == f"{fig_cpu_time_dir}/cpu_time_1_threads_normal":
            # swap S-Classifier is next to P-Classifier
            algorithms_list = df_cpu_time.index.tolist()
            algorithms_list[1], algorithms_list[2] = algorithms_list[2], algorithms_list[1]
            df_cpu_time = df_cpu_time.reindex(algorithms_list)
            df_cpu_time = df_cpu_time.reindex(algorithms_list)
        df_cpu_time.to_csv(f"{filename_prefix}.csv", index=True, header=True)
        print(df_cpu_time)

        # Print improvement ratios
        print_improvement_ratio(df_cpu_time)

        # Create bar chart
        create_bar_chart(df_cpu_time, True, 'CPU time (seconds)', 'Datasets', 'Algorithms', 'lower right',
                         f"{filename_prefix}.png")

if method == 0 or method == 3:
    print("Print hash distribution information")
    os.makedirs(fig_hash_performance_dir, exist_ok=True)


    def get_face_hash_distribution(filename):
        # Read the yaml file
        with open(filename, "r") as file:
            content = yaml.load(file, Loader=yaml.FullLoader)[0]
            experiments = content["experiments"]
            face_hash_distribution = experiments["face-hash-distribution"]
            return face_hash_distribution


    face_hash_distribution_data = {}
    for dataset in datasets:
        dataset_name = get_dataset_name(dataset)
        hash_distribution_file = f"{data_hash_performance_dir}/{dataset_name}_hash_distribution.yaml"
        with open(hash_distribution_file, "r") as file:
            content = yaml.load(file, Loader=yaml.FullLoader)[0]
            experiments = content["experiments"]
            face_hash_distribution_data[dataset_name] = experiments["face-hash-distribution"]

    # Plot settings
    fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

    # Iterate over the data and plot
    for i, (key, values) in enumerate(face_hash_distribution_data.items()):
        print(f"Processing dataset: {key}")
        # Plot Hash Functions' Distribution
        line_width = 1
        for hash_function_name in reversed(hash_function_names[1:]):
            hash_values = values[hash_function_name]
            axs[i].plot(hash_values.keys(), hash_values.values(), label=hash_function_name, alpha=0.6,
                        linewidth=line_width)
            line_width += 0.5

        axs[i].set_xlabel('Number of faces per hash', fontsize=axis_label_fontsize)
        axs[i].set_ylabel('Count', fontsize=axis_label_fontsize)
        # axs[i].set_title(f"{key} - Hash Functions' Distribution", fontsize=title_fontsize)
        axs[i].set_title(f"{key}", fontsize=title_fontsize)

        axs[i].legend(title='Hash functions', loc='upper right', fontsize=legend_fontsize)
        axs[i].grid(True, linewidth=0.25)

        # Prepare data for the boxplots
        # boxplot_data = [
        #     [int(num_faces) for num_faces, count in values[hash_function_name].items() for _ in range(count)]
        #     for hash_function_name in hash_function_names[1:]
        # ]
        # # Create the boxplot for the current dataset
        # axs[i].boxplot(
        #     boxplot_data,
        #     labels=hash_function_names[1:],
        #     vert=True,  # Set boxplots to vertical
        # )
        # axs[i].set_xlabel('Hash Functions', fontsize=axis_label_fontsize)
        # axs[i].set_ylabel('Number of Faces', fontsize=axis_label_fontsize)
        # axs[i].set_title(f"{key}", fontsize=title_fontsize)
        # axs[i].grid(True)

    filename = f"{fig_hash_performance_dir}/hash_distributions.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    print("Print cache misses information")


    def get_cache_misses_info(filename):
        # Read the file content
        with open(filename, 'r') as file:
            text = file.read()
            # Use a regular expression to find the cache-misses value
            match = re.search(r'([\d,]+)\s+cache-misses', text)
            return int(match.group(1).replace(",", "")) if match else 0


    cache_misses_data = {}
    for dataset in datasets:
        dataset_name = get_dataset_name(dataset)
        cache_misses_data[dataset_name] = {}
        # cache misses for reading the dataset
        dataset_cache_misses_file = f"{data_hash_performance_dir}/{dataset_name}_cache_misses.txt"
        dataset_cache_misses = get_cache_misses_info(dataset_cache_misses_file)
        for algo in vtkm_algorithms:
            for hash_function in hash_functions[1:]:
                algorithm_name = f"{algorithms_names[algo]}-{hash_function_names[hash_function]}"
                algorithm_cache_misses_file = f"{data_hash_performance_dir}/{dataset_name}_{algorithm_name}_cache_misses.txt"
                total_cache_misses = get_cache_misses_info(algorithm_cache_misses_file)
                algo_cache_misses = total_cache_misses - dataset_cache_misses
                cache_misses_data[dataset_name][algorithm_name] = algo_cache_misses

    # Convert the data to a pandas DataFrame
    df_cache_misses = pd.DataFrame(cache_misses_data)
    df_cache_misses.index.name = 'Algorithm'
    df_cache_misses.to_csv(f"{fig_hash_performance_dir}/cache_misses.csv", index=True, header=True)
    print(df_cache_misses)

    # Print improvement ratios
    print_improvement_ratio(df_cache_misses)

    # create bar chart
    create_bar_chart(df_cache_misses, True, 'CPU L3 cache misses (count)', 'Datasets', 'Algorithms', 'lower right',
                     f"{fig_hash_performance_dir}/cache_misses.png")

if method == 0 or method == 4:
    print("Print Speed-up information")
    os.makedirs(fig_speed_up_dir, exist_ok=True)

    cpu_time = {}
    biggest_datasets_names = [get_dataset_name(dataset) for dataset in biggest_datasets]
    for dataset_name in biggest_datasets_names:
        cpu_time[dataset_name] = {}
        for power in range(0, max_number_of_threads_power_of_2 + 1):
            threads = int(math.pow(2, power))
            cpu_time[dataset_name][threads] = get_run_time_info(
                f"{data_speed_up_dir}/{dataset_name}_{threads}_threads.yaml")

    # Convert the data to a pandas DataFrame
    df_speed_up_1_dataset = pd.DataFrame(cpu_time[biggest_datasets_names[0]]).transpose()
    df_speed_up_1_dataset.index.name = 'Threads'
    df_speed_up_1_dataset.to_csv(f"{fig_speed_up_dir}/{biggest_datasets_names[0]}_speed_up.csv", index=True,
                                 header=True)
    df_speed_up_2_dataset = pd.DataFrame(cpu_time[biggest_datasets_names[1]]).transpose()
    df_speed_up_2_dataset.index.name = 'Threads'
    df_speed_up_2_dataset.to_csv(f"{fig_speed_up_dir}/{biggest_datasets_names[1]}_speed_up.csv", index=True,
                                 header=True)

    speed_up_dataframes = [df_speed_up_1_dataset, df_speed_up_2_dataset]

    for dataset_name, df_speed_up in zip(biggest_datasets_names, speed_up_dataframes):
        print(df_speed_up)
        # Create the figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Loop through each dataset and plot its values across threads
        for algorithm_name in df_speed_up.columns:
            # Extract time for the dataset
            values = df_speed_up[algorithm_name].values
            # speed_up_values = values[0] / values # speed-up over itself
            speed_up_values = df_speed_up.iloc[0].min() / values  # speed-up over the best sequential time

            # # Plot each dataset's speed-up as a line for each thread
            # ax.plot(df_speed_up.index, speed_up_values, marker='o', label=algorithm_name, alpha=0.8,
            #         color=algorithm_colors[algorithm_name])

            # Plot the solid line up to the second-to-last point, for actual threads
            ax.plot(df_speed_up.index[:-1], speed_up_values[:-1], marker='o', label=algorithm_name,
                    alpha=0.8, color=algorithm_colors[algorithm_name])

            # Plot the last point with a dashed line, for hyper threads
            ax.plot(df_speed_up.index[-2:], speed_up_values[-2:], marker='o', linestyle='--',
                    color=algorithm_colors[algorithm_name])

        # Set the x-axis to a logarithmic scale
        ax.set_xscale('log', base=2)
        ax.set_xticks(df_speed_up.index)
        ax.set_xticklabels(df_speed_up.index)

        # Add labels and title
        ax.set_ylabel('Speed-up', fontsize=axis_label_fontsize)
        ax.set_xlabel('Threads', fontsize=axis_label_fontsize)
        # ax.set_title('Speed-Up by Threads', fontsize=title_fontsize)

        # Add legend
        ax.legend(title='Algorithms', loc='upper left', fontsize=legend_fontsize)

        # Show grid lines for clarity
        ax.grid()

        filename = f"{fig_speed_up_dir}/{dataset_name}_speed_up.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')

if method == 0 or method == 5:
    print("Print GPU Time information")
    os.makedirs(fig_gpu_time_dir, exist_ok=True)

    gpu_time_data_normal = {}
    gpu_time_data_random = {}
    for dataset in datasets:
        dataset_name = get_dataset_name(dataset)
        gpu_time_data_normal[dataset_name] = {}
        gpu_time_data_random[dataset_name] = {}
        for algo in vtkm_algorithms:
            for hash_function in hash_functions[1:]:
                algorithm_name = f"{algorithms_names[algo]}-{hash_function_names[hash_function]}"

                normal_file = f"{data_gpu_time_dir}/{dataset_name}_{algorithms_names[algo]}-{hash_function_names[hash_function]}_normal.yaml"
                run_time_normal = get_run_time_info(normal_file)
                gpu_time_data_normal[dataset_name][algorithm_name] = run_time_normal[
                    algorithm_name] if run_time_normal is not None else np.nan

                random_file = f"{data_gpu_time_dir}/{dataset_name}_{algorithms_names[algo]}-{hash_function_names[hash_function]}_random_{random_seed}.yaml"
                run_time_random = get_run_time_info(random_file)
                gpu_time_data_random[dataset_name][algorithm_name] = run_time_random[
                    algorithm_name] if run_time_random is not None else np.nan

    filename_prefixes = [f"{fig_gpu_time_dir}/gpu_time_normal", f"{fig_gpu_time_dir}/gpu_time_random_{random_seed}"]
    df_gpu_time_data = [gpu_time_data_normal, gpu_time_data_random]
    for filename_prefix, gpu_time_data in zip(filename_prefixes, df_gpu_time_data):
        print(f"Processing file: {filename_prefix}")
        # Convert the data to a pandas DataFrame
        df_gpu_time = pd.DataFrame(gpu_time_data)
        df_gpu_time.index.name = 'Algorithm'
        df_gpu_time.to_csv(f"{filename_prefix}.csv", index=True, header=True)
        print(df_gpu_time)

        # Print improvement ratios
        print_improvement_ratio(df_gpu_time)

        # Create bar chart
        create_bar_chart(df_gpu_time, False, 'GPU time (seconds)', 'Datasets', 'Algorithms', 'center right',
                         f"{filename_prefix}.png")
