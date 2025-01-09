import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy import stats

# Set matplotlib parameters for publication-quality plots
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

def get_simulation_params(base_path):
    """Read simulation parameters from the first .out file found in any subdirectory."""
    for directory in Path(base_path).glob('N1_n*_omp*'):  # Modified pattern for OMP
        for file in directory.glob('*.out'):
            params = {
                'num_particles': None,
                'num_particles_per_node': None,
                'max_particles': None,
                'max_depth': None,
                'dist': None,
                'iterations': None,
                'omp_threads': None  # Added OMP threads parameter
            }
            
            # Extract OMP threads from directory name
            omp_threads = int(directory.name.split('_omp')[1])
            params['omp_threads'] = omp_threads
            
            with open(file, 'r') as f:
                for line in f:
                    if "Option '-num_particles=" in line:
                        params['num_particles_per_node'] = int(line.split('=')[1].split("'")[0])
                    elif "Option '-num_particles_tot=" in line:
                        params['num_particles'] = int(line.split('=')[1].split("'")[0])
                    elif "Option '-max_particles=" in line:
                        params['max_particles'] = int(line.split('=')[1].split("'")[0])
                    elif "Option '-max_depth=" in line:
                        params['max_depth'] = int(line.split('=')[1].split("'")[0])
                    elif "Option '-dist=" in line:
                        params['dist'] = line.split('=')[1].split("'")[0]
                    elif "Option '-iterations=" in line:
                        params['iterations'] = int(line.split('=')[1].split("'")[0])
            
            return params
    
    return None

def parse_timing_file(filepath, iterations=1):
    data = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract node and OMP thread count from directory name
    dir_name = filepath.parent.name
    n_nodes = int(dir_name.split('_n')[1].split('_')[0])
    omp_threads = int(dir_name.split('_omp')[1])
    
    reading_totals = False
    reading_averages = False
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('='): 
            continue
            
        if 'Wall tot' in line:
            reading_totals = True
            reading_averages = False
            continue
            
        if 'Wall max' in line:
            reading_totals = False
            reading_averages = True
            continue
            
        parts = line.split()
        operation = parts[0].strip('.')
        
        if reading_totals and len(parts) >= 3:
            data[operation] = {
                'nodes': n_nodes,
                'omp_threads': omp_threads,  # Added OMP threads
                'wall_tot': float(parts[-1]) / iterations
            }
        elif reading_averages and len(parts) >= 5:
            if operation not in data:
                data[operation] = {'nodes': n_nodes, 'omp_threads': omp_threads}
            data[operation].update({
                'wall_max': float(parts[-3]) / iterations,
                'wall_min': float(parts[-2]) / iterations,
                'wall_avg': float(parts[-1]) / iterations
            })
    
    return data

def collect_all_timing_data(base_path, iterations=1):
    all_data = []
    
    # Modified pattern to match OMP directories
    for directory in sorted(Path(base_path).glob('N1_n*_omp*')):
        timing_file = directory / 'timings.dat'
        if timing_file.exists():
            data = parse_timing_file(timing_file, iterations)
            all_data.append(data)
    
    return all_data

def plot_omp_timing_analysis(df, sim_params, operations=None, figsize=(15, 8), dpi=300):
    """Create bar plot showing execution time for different OMP thread configurations."""
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=figsize, dpi=dpi)
    
    if operations is not None:
        df_filtered = df[df['operation'].isin(operations)]
    else:
        df_filtered = df

    # Sort by number of threads for consistent x-axis ordering
    df_filtered = df_filtered.sort_values('omp_threads')
    
    # Create labels that show both node count and thread count
    x_labels = [f'n{row.nodes}_t{row.omp_threads}' for _, row in df_filtered.drop_duplicates(['nodes', 'omp_threads']).iterrows()]
    
    # Set up the bar positions
    num_configs = len(df_filtered['omp_threads'].unique())
    num_ops = len(df_filtered['operation'].unique())
    width = 0.8 / num_ops
    
    # Create bars for each operation
    for idx, operation in enumerate(sorted(df_filtered['operation'].unique())):
        operation_data = df_filtered[df_filtered['operation'] == operation]
        positions = np.arange(len(x_labels)) + idx * width - (num_ops-1) * width/2
        
        plt.bar(positions, 
               operation_data['wall_avg'],
               width=width,
               label=operation)
        
        # Add value labels on top of each bar
        for pos, val in zip(positions, operation_data['wall_avg']):
            plt.text(pos, val, f'{val:.2f}s', 
                    ha='center', va='bottom', rotation=90,
                    fontsize=8)
    
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
    
    title = (f'OpenMP Execution Time Analysis\n'
            f'N={sim_params["num_particles"]}, '
            f'max_part={sim_params["max_particles"]}, '
            f'depth={sim_params["max_depth"]}')
    
    plt.title(title)
    plt.xlabel('Configuration (nodes_threads)')
    plt.ylabel('Wall Time [s]')
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    filename = 'omp_timing_analysis.png'
    if operations:
        filename = f'omp_timing_{"_".join(operations)}.png'
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    plt.close()

def main():
    base_path = '.'
    
    # Clean up existing plots directory
    if os.path.exists('plots'):
        import shutil
        shutil.rmtree('plots')
    
    # Get simulation parameters
    sim_params = get_simulation_params(base_path)
    
    # Collect timing data
    all_data = collect_all_timing_data(base_path, sim_params.get('iterations', 1))
    
    # Create DataFrame
    df = pd.DataFrame([
        {'operation': op, **metrics}
        for data_dict in all_data
        for op, metrics in data_dict.items()
    ])
    
    # Generate plots
    print("Available operations:", sorted(df['operation'].unique()))
    
    # Create various plots for different operations
    operations_to_analyze = [
        ['orthotree_build'],
        ['build_tree'],
        ['build_tree_from_oct'],
    ]
    
    for ops in operations_to_analyze:
        plot_omp_timing_analysis(df, sim_params, operations=ops)
    
    # Also create one plot with all operations
    plot_omp_timing_analysis(df, sim_params)
    
    # Save processed data
    df.to_csv('omp_timing_analysis.csv', index=False)
    
    # Print summary statistics
    print("\nSummary of timing data:")
    print(df.groupby(['operation', 'omp_threads'])['wall_avg'].describe())

if __name__ == '__main__':
    main()
