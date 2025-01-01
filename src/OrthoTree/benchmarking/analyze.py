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

def plot_scaling_analysis(df, sim_params, operations=None, figsize=(12, 8), dpi=300, ideal_scaling=True):
    """
    Create a publication-quality scatter plot with regression analysis for wall time scaling.
    
    Args:
        df: DataFrame containing the timing data
        operations: List of operation names to plot. If None, plots all operations
        figsize: Tuple specifying figure dimensions (width, height) in inches
        dpi: Resolution of the output figure
        ideal_scaling: If True, adds an ideal scaling line for comparison
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    # Create a directory for plots if it doesn't exist
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Set the style for publication-quality plots
    plt.style.use('seaborn-v0_8-paper')
    
    # Create figure with specified size and DPI
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Filter the DataFrame for specified operations
    if operations is not None:
        df_filtered = df[df['operation'].isin(operations)]
    else:
        df_filtered = df
    
    # Color palette for better distinction between operations
    colors = sns.color_palette("deep", n_colors=len(df_filtered['operation'].unique()))
    
    # Plot each operation with error analysis
    for idx, operation in enumerate(df_filtered['operation'].unique()):
        operation_data = df_filtered[df_filtered['operation'] == operation]
        x = operation_data['nodes']
        y = operation_data['wall_avg']
        
        # Normalize times relative to the smallest number of nodes
        base_nodes = x.min()
        base_time = y[x == base_nodes].iloc[0]
        y_normalized = y / base_time
        
        # Plot scatter points with error bars if available
        if 'wall_std' in operation_data.columns:
            plt.errorbar(x, y, yerr=operation_data['wall_std'], 
                        fmt='o', label=operation, color=colors[idx],
                        capsize=3, capthick=1, elinewidth=1, markersize=6)
        else:
            plt.scatter(x, y, label=operation, color=colors[idx], s=50)
        
        # Fit power law: y = ax^b
        log_x = np.log(x)
        log_y = np.log(y)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        
        # Generate points for the fit line
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = np.exp(intercept) * x_fit**slope
        
        # Plot fit line
        plt.plot(x_fit, y_fit, '--', color=colors[idx], alpha=0.7,
                label=f'{operation}\n(scaling factor={slope:.2f}, R²={r_value**2:.3f})')
    
    # Add ideal scaling line if requested
    if ideal_scaling:
        min_nodes = df_filtered['nodes'].min()
        max_nodes = df_filtered['nodes'].max()
        base_time = df_filtered[df_filtered['nodes'] == min_nodes]['wall_avg'].min()
        x_ideal = np.linspace(min_nodes, max_nodes, 100)
        y_ideal = base_time * min_nodes / x_ideal
        plt.plot(x_ideal, y_ideal, 'k:', label='Ideal scaling (1/x)', alpha=0.5)
    
    # Customize the plot
    ax.set_yscale('log')  # Only y-axis in log scale
    
    # Set x-axis ticks to match the actual node numbers
    node_values = sorted(df_filtered['nodes'].unique())
    plt.xticks(node_values, [str(x) for x in node_values])
    
    # Set grid with minor gridlines
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.15)
    
    # Customize labels and title
        # Get the simulation parameters for the title
    # Create title based on available parameters
    if sim_params.get('num_particles_per_node'):
        title = (f'Operation Time Breakdown by Node Count\n'
                f'(part_per_node={sim_params["num_particles_per_node"]}, '
                f'max_part={sim_params["max_particles"]}, '
                f'depth={sim_params["max_depth"]}, '
                f'dist={sim_params["dist"]})')
    else:
        title = (f'Operation Time Breakdown by Node Count\n'
                f'(N={sim_params["num_particles"]}, '
                f'max_part={sim_params["max_particles"]}, '
                f'depth={sim_params["max_depth"]}, '
                f'dist={sim_params["dist"]})')
    
    plt.title(title)
    plt.xlabel('Number of CPUs', fontsize=12)
    plt.ylabel('Wall Time [s]', fontsize=12)
    
    # Customize legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    
    # Add text box with additional information
    textstr = '\n'.join([
        'Scaling Analysis:',
        'y = ax^b where b is the scaling factor',
        'Ideal scaling: b = -1',
        'b → -1: Better scaling',
        'b → 0: Poor scaling'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.text(1.05, 0.1, textstr, transform=ax.transAxes, fontsize=9,
             verticalalignment='bottom', bbox=props)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot with high quality
    if operations is not None:
        filename = f'scaling_analysis_{"_".join(operations)}.png'
    else:
        filename = 'scaling_analysis.png'
    
    plt.savefig(os.path.join(plot_dir, filename), 
                dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.close()

def plot_operation_breakdown(df, sim_params, operations=None):
    """
    Create a stacked bar chart for specified operations.
    
    Args:
        df: DataFrame containing the timing data
        operations: List of operation names to plot. If None, plots all operations.
    """
    # Create a directory for plots if it doesn't exist
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(15, 8))
    
    # Filter the DataFrame for specified operations
    if operations is not None:
        df_filtered = df[df['operation'].isin(operations)]
    else:
        df_filtered = df
    
    # Pivot the filtered data
    pivot_data = df_filtered.pivot(index='nodes', columns='operation', values='wall_avg')
    pivot_data.plot(kind='bar', stacked=True)
    
    # Create title based on available parameters
    if sim_params.get('num_particles_per_node'):
        title = (f'Operation Time Breakdown by Node Count\n'
                f'(part_per_node={sim_params["num_particles_per_node"]}, '
                f'max_part={sim_params["max_particles"]}, '
                f'depth={sim_params["max_depth"]}, '
                f'dist={sim_params["dist"]})')
    else:
        title = (f'Operation Time Breakdown by Node Count\n'
                f'(N={sim_params["num_particles"]}, '
                f'max_part={sim_params["max_particles"]}, '
                f'depth={sim_params["max_depth"]}, '
                f'dist={sim_params["dist"]})')
    
    plt.title(title)
    plt.xlabel('Number of CPUs')
    plt.ylabel('Wall Time (s)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot in the plots directory
    if operations is not None:
        plt.savefig(os.path.join(plot_dir, 'operation_breakdown_{}.png'.format('_'.join(operations))))
    else:
        plt.savefig(os.path.join(plot_dir, 'operation_breakdown.png'))

    plt.close()

def plot_operation_grouped(df, sim_params, operations=None, exclude_operations=None):
    """
    Create a grouped bar chart for specified operations, excluding certain operations if needed.
    
    Args:
        df: DataFrame containing the timing data.
        sim_params: Dictionary containing simulation parameters.
        operations: List of operation names to plot. If None, plots all operations.
        exclude_operations: List of operation names to exclude from the plot.
    """
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    # Create a more specific filename based on the operations
    filename = 'operation_grouped'
    if operations:
        filename += '_' + '_'.join(operations)
    if exclude_operations:
        filename += '_exclude_' + '_'.join(exclude_operations)
    filename += '.png'

    plt.figure(figsize=(15, 8))
    
    # Filter the DataFrame for specified operations
    if operations is not None:
        df_filtered = df[df['operation'].isin(operations)]
    else:
        df_filtered = df
    
    # Exclude specified operations
    if exclude_operations is not None:
        df_filtered = df_filtered[~df_filtered['operation'].isin(exclude_operations)]
    
    # Pivot the filtered data
    pivot_data = df_filtered.pivot(index='nodes', columns='operation', values='wall_avg')
    
    # Plot using a grouped bar chart
    ax = pivot_data.plot(kind='bar', width=0.8)
    
    # Create title based on available parameters
    if sim_params.get('num_particles_per_node'):
        title = (f'Operation Time Breakdown by Node Count\n'
                f'(part_per_node={sim_params["num_particles_per_node"]}, '
                f'max_part={sim_params["max_particles"]}, '
                f'depth={sim_params["max_depth"]}, '
                f'dist={sim_params["dist"]})')
    else:
        title = (f'Operation Time Breakdown by Node Count\n'
                f'(N={sim_params["num_particles"]}, '
                f'max_part={sim_params["max_particles"]}, '
                f'depth={sim_params["max_depth"]}, '
                f'dist={sim_params["dist"]})')
    
    plt.title(title)
    plt.xlabel('Number of CPUs')
    plt.ylabel('Wall Time (s)')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

def plot_operation_with_breakdown(df, sim_params, main_operation, operations=None, exclude_operations=None):
    """
    Create a bar chart showing a main operation alongside the breakdown of other operations.
    
    Args:
        df: DataFrame containing the timing data.
        sim_params: Dictionary containing simulation parameters.
        main_operation: String specifying which operation to show as the main bar.
        operations: List of operation names to plot. If None, plots all operations except main_operation.
        exclude_operations: List of operation names to exclude from the plot.
    """
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    # Create a more specific filename based on the operations
    filename = f'operation_bar_{main_operation}'
    if operations:
        filename += '_' + '_'.join(operations)
    if exclude_operations:
        filename += '_exclude_' + '_'.join(exclude_operations)
    filename += '.png'

    plt.figure(figsize=(15, 8))
    
    # Filter the DataFrame for specified operations
    if operations is not None:
        other_operations = [op for op in operations if op != main_operation]
        df_filtered = df[df['operation'].isin([main_operation] + other_operations)]
    else:
        other_operations = [op for op in df['operation'].unique() if op != main_operation]
        df_filtered = df[df['operation'].isin([main_operation] + other_operations)]
    
    # Exclude specified operations
    if exclude_operations is not None:
        df_filtered = df_filtered[~df_filtered['operation'].isin(exclude_operations)]
        other_operations = [op for op in other_operations if op not in exclude_operations]
    
    # Create pivot table for the data
    pivot_data = df_filtered.pivot(index='nodes', columns='operation', values='wall_avg')
    
    # Plot main operation as individual bars
    ax = plt.gca()
    x = range(len(pivot_data.index))
    main_times = pivot_data[main_operation]
    ax.bar(x, main_times, width=0.4, label=main_operation)
    
    # Plot stacked bars for other operations
    other_data = pivot_data[other_operations]
    left = [0] * len(pivot_data.index)
    for column in other_data.columns:
        ax.bar([i + 0.5 for i in x], other_data[column], width=0.4,
               bottom=left, label=column)
        left = [l + v for l, v in zip(left, other_data[column])]
    
    # Set x-axis labels
    plt.xticks([i + 0.25 for i in x], pivot_data.index)
    
    # Create title based on available parameters
    if sim_params.get('num_particles_per_node'):
        title = (f'Operation Time Breakdown by Node Count\n'
                f'(part_per_node={sim_params["num_particles_per_node"]}, '
                f'max_part={sim_params["max_particles"]}, '
                f'depth={sim_params["max_depth"]}, '
                f'dist={sim_params["dist"]})')
    else:
        title = (f'Operation Time Breakdown by Node Count\n'
                f'(N={sim_params["num_particles"]}, '
                f'max_part={sim_params["max_particles"]}, '
                f'depth={sim_params["max_depth"]}, '
                f'dist={sim_params["dist"]})')
    
    plt.title(title)
    plt.xlabel('Number of CPUs')
    plt.ylabel('Wall Time (s)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

def plot_operation_speedup(df, sim_params, operations=None):
    """
    Create a speedup plot for specified operations.
    
    Args:
        df: DataFrame containing the timing data.
        sim_params: Dictionary containing simulation parameters.
        operations: List of operation names to plot. If None, plots all operations.
    """
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(15, 8))
    
    # Filter the DataFrame for specified operations
    if operations is not None:
        df_filtered = df[df['operation'].isin(operations)]
    else:
        df_filtered = df
    
    # Pivot the filtered data
    pivot_data = df_filtered.pivot(index='nodes', columns='operation', values='wall_avg')
    
    # Calculate speedup relative to the smallest number of nodes
    base_nodes = pivot_data.index.min()
    base_times = pivot_data.loc[base_nodes]
    
    # Correct speedup calculation: base_time / time_with_n_nodes
    speedup_data = base_times / pivot_data  # This is the key change
    
    # Plot using a line plot with markers
    ax = speedup_data.plot(marker='o', markersize=8, linewidth=2)
    
    # Add ideal speedup line
    nodes = pivot_data.index
    ideal_speedups = [n/base_nodes for n in nodes]
    plt.plot(nodes, ideal_speedups, 'k--', label='Ideal Speedup', alpha=0.5)
    
    # Create title based on available parameters
    if sim_params.get('num_particles_per_node') not in [None, 0]:
        title = (f'Operation Speedup Analysis\n'
                f'(part_per_node={sim_params["num_particles_per_node"]}, '
                f'max_part={sim_params["max_particles"]}, '
                f'depth={sim_params["max_depth"]}, '
                f'dist={sim_params["dist"]})')
    else:
        title = (f'Operation Speedup Analysis\n'
                f'(N={sim_params["num_particles"]}, '
                f'max_part={sim_params["max_particles"]}, '
                f'depth={sim_params["max_depth"]}, '
                f'dist={sim_params["dist"]})')
    
    plt.title(title)
    plt.xlabel('Number of CPUs')
    plt.ylabel(f'Speedup (relative to {base_nodes} nodes)')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set reasonable y-axis limits
    plt.ylim(bottom=0)
    
    # Create a more specific filename based on the operations
    filename = 'operation_speedup'
    if operations:
        filename += '_' + '_'.join(operations)
    filename += '.png'
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()


def main():
    # Assuming you're in the parent directory containing all N*_n* directories
    base_path = '.'

    # Check if plots directory exists and delete it
    if os.path.exists('plots'):
        import shutil
        shutil.rmtree('plots')
    
    # Get simulation parameters once at the start
    sim_params = get_simulation_params(base_path)
    
    # Collect all timing data
    all_data = collect_all_timing_data(base_path, sim_params.get('iterations', 1))
    
    # Create a pandas DataFrame
    df = create_timing_dataframe(all_data)
    
    # Generate plots
    print("Available operations:", sorted(df['operation'].unique()))

    plot_operation_with_breakdown(df, sim_params, 'orthotree_build', operations=['build_tree', 'aid_list'])

    plot_operation_with_breakdown(df, sim_params, 'build_tree', operations=['block_partition', 'build_tree_from_oct'])

    plot_operation_with_breakdown(df, sim_params, 'block_partition', operations=['innitfromoctants'])

    # plot_operation_breakdown(df, sim_params, operations=['orthotree_build'])
    plot_operation_breakdown(df, sim_params, operations=['build_tree'])
    # plot_operation_breakdown(df, sim_params, operations=['block_partition'])
    # plot_operation_breakdown(df, sim_params, operations=['partition'])
    # plot_operation_breakdown(df, sim_params, operations=['getNumParticlesInOc'])

    # plot_scaling_analysis(df, sim_params, operations=['orthotree_build'])
    # plot_scaling_analysis(df, sim_params, operations=['build_tree'])
    # plot_scaling_analysis(df, sim_params, operations=['getNumParticlesInOc'])
    # plot_scaling_analysis(df, sim_params, operations=['block_partition'])
    # plot_scaling_analysis(df, sim_params, operations=['partition'])

    # plot_operation_speedup(df, sim_params, operations=['orthotree_build'])
    # plot_operation_speedup(df, sim_params, operations=['build_tree'])
    # plot_operation_speedup(df, sim_params, operations=['getNumParticlesInOc'])
    # plot_operation_speedup(df, sim_params, operations=['block_partition'])
    # plot_operation_speedup(df, sim_params, operations=['partition'])
    

    # Save the processed data
    df.to_csv('timing_analysis.csv', index=False)
    
    # Print summary statistics
    print("\nSummary of timing data:")
    print(df.groupby('operation')['wall_avg'].describe())

if __name__ == '__main__':
    main()
