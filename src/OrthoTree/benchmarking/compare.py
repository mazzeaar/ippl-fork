import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
import seaborn as sns
from pathlib import Path

def get_simulation_params(base_path):
    """Read simulation parameters from the first .out file found in any subdirectory."""
    for directory in Path(base_path).glob('N[123]_n*'):
        for file in directory.glob('*.out'):
            params = {
                'num_particles': None,
                'num_particles_per_node': None,
                'max_particles': None,
                'max_depth': None,
                'dist': None,
                'iterations': None
            }
            
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
            
            # Return the first set of parameters found
            return params
    
    return None

def parse_timing_file(filepath, iterations=1):
    # Initialize data storage
    data = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Extract number of nodes from directory name
    n_nodes = int(filepath.parent.name.split('_n')[1])
    
    # Parse the timing data
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
                'wall_tot': float(parts[-1]) / iterations
            }
        elif reading_averages and len(parts) >= 5:
            if operation not in data:
                data[operation] = {'nodes': n_nodes}
            data[operation].update({
                'wall_max': float(parts[-3]) / iterations,
                'wall_min': float(parts[-2]) / iterations,
                'wall_avg': float(parts[-1]) / iterations
            })
    
    return data

def collect_all_timing_data(base_path, iterations=1):
    all_data = []
    
    # Walk through all N1_* and N2_* directories
    for directory in sorted(Path(base_path).glob('N[123456789]_n*')):
        timing_file = directory / 'timings.dat'
        if timing_file.exists():
            data = parse_timing_file(timing_file, iterations)
            all_data.append(data)
    
    return all_data

def create_timing_dataframe(all_data):
    # Flatten the data structure into a format suitable for pandas
    rows = []
    for data_dict in all_data:
        for operation, metrics in data_dict.items():
            row = {'operation': operation, **metrics}
            rows.append(row)
    
    return pd.DataFrame(rows)

def compare_operation(operation, df1, df2, df1_label="Old", df2_label="New", title=None):
    """
    Compare a specific operation between two dataframes and create a plot.
    
    Parameters:
    -----------
    operation : str
        The operation to compare
    df1 : pandas.DataFrame
        First dataframe to compare
    df2 : pandas.DataFrame
        Second dataframe to compare
    df1_label : str
        Label for the first dataframe in the plot legend
    df2_label : str
        Label for the second dataframe in the plot legend
    title : str, optional
        Custom title for the plot. If None, uses the operation name
    """
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Filter data for the specific operation
    data1 = df1[df1['operation'] == operation]
    data2 = df2[df2['operation'] == operation]
    
    # Create the plot
    plt.plot(data1['nodes'], data1['wall_tot'], 'o-', label=df1_label, linewidth=2, markersize=10)
    plt.plot(data2['nodes'], data2['wall_tot'], 's-', label=df2_label, linewidth=2, markersize=10)
    
    # Customize the plot
    plt.xlabel('Number of CPUs')
    plt.ylabel('Wall Time (s)')
    plt.title(title if title else f'Comparison of {operation}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save the plot
    plt.savefig(f'plots/{operation}_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()


def main():
    # Assuming you're in the parent directory containing all N*_n* directories
    base_path = '.'
    weak_scaling_old_path = './weak_scaling_spiral_it5_old'
    weak_scaling_new_path = './weak_scaling_spiral_it5_new'
    strong_scaling_old_path = './strong_scaling_spiral_it5_old'
    strong_scaling_new_path = './strong_scaling_spiral_it5_new'

    # Check if plots directory exists and delete it
    if os.path.exists('plots'):
        import shutil
        shutil.rmtree('plots')
    
    # Get simulation parameters once at the start
    sim_param_weak = get_simulation_params(weak_scaling_old_path)
    sim_param_strong = get_simulation_params(strong_scaling_old_path)

    
    # Collect all timing data
    data_weak_old = collect_all_timing_data(weak_scaling_old_path, sim_param_weak.get('iterations', 1))
    data_weak_new = collect_all_timing_data(weak_scaling_new_path, sim_param_weak.get('iterations', 1))
    data_strong_old = collect_all_timing_data(strong_scaling_old_path, sim_param_strong.get('iterations', 1))
    data_strong_new = collect_all_timing_data(strong_scaling_new_path, sim_param_strong.get('iterations', 1))
    
    # Create a pandas DataFrame
    df_weak_old = create_timing_dataframe(data_weak_old)
    df_weak_new = create_timing_dataframe(data_weak_new)
    df_strong_old = create_timing_dataframe(data_strong_old)
    df_strong_new = create_timing_dataframe(data_strong_new)

    
    # Generate plots
    print("Available operations:", sorted(df_weak_new['operation'].unique()))



if __name__ == '__main__':
    main()
