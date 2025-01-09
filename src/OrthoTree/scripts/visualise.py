import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools

def read_data_for_all_processors(data_folder):
    processor_data = []
    num_processors = len([f for f in os.listdir(data_folder) if f.startswith("octants") and f.endswith(".txt")])

    for i in range(num_processors):
        octants_file = os.path.join(data_folder, f"octants{i}.txt")
        particles_file = os.path.join(data_folder, f"particles{i}.txt")
        
        if not os.path.isfile(octants_file) or not os.path.isfile(particles_file):
            print(f"Warning: Missing file(s) for processor {i}")
            continue

        octants = read_octants_file(octants_file)
        particles = read_particles_file(particles_file)
        dim = len(particles[0]["coord"]) if particles else 2  # Default to 2D if no particles
        # print("OCTANTS:", octants)
        # print("PARTICLES:", particles)
        # print("DIM:", dim)
        processor_data.append((octants, particles, dim))

    return processor_data

# Reads a coordinate in the form (x, y, z) or (x, y)
def read_coordinate(coord_str):
    coords = coord_str.strip("()").split(", ")
    if len(coords) == 2:
        return float(coords[0]), float(coords[1])  # Return 2D coordinate
    elif len(coords) == 3:
        return float(coords[0]), float(coords[1]), float(coords[2])  # Return 3D coordinate
    else:
        raise ValueError("Unexpected coordinate format")

def read_octant_line(line):
    parts = line.split(' ')
    morton_code = int(parts[0])
    if len(parts) >= 14:  # 3D case with (x, y, z) coordinates
        coord_min = read_coordinate(f"({parts[2]}, {parts[4]}, {parts[6]})")
        coord_max = read_coordinate(f"({parts[9]}, {parts[11]}, {parts[13]})")
    elif len(parts) >= 10:  # 2D case with (x, y) coordinates
        coord_min = read_coordinate(f"({parts[2]}, {parts[4]})")
        coord_max = read_coordinate(f"({parts[7]}, {parts[9]})")
    else:   
        raise ValueError("Unexpected format in octant line: insufficient data")

    return {
        "morton_code": morton_code,
        "coord_min": coord_min,
        "coord_max": coord_max
    }

def read_octants_file(file_path):
    octants = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  
            octant = read_octant_line(line)
            octants.append(octant)
    return octants

def read_particle_line(line):
    parts = line.split(' ')
    morton_code = int(parts[0])
    if len(parts) >= 7:
        coord = read_coordinate(f"({parts[2]}, {parts[4]}, {parts[6]})")
    elif len(parts) >= 4:
        coord = read_coordinate(f"({parts[2]}, {parts[4]})")
    else:
        raise ValueError("Unexpected format in particle line: insufficient data")
    
    return {
        "morton_code": morton_code,
        "coord": coord
    }

def read_particles_file(file_path):
    particles = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  
            particle = read_particle_line(line)
            particles.append(particle)
    return particles

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_bounds(ax, coord_min, coord_max, morton_code=None, color='blue', alpha=0.3, fill=True, border_width=1.5):
    """
    Draw bounding boxes for octants in either 2D or 3D, with proper border width and optional fill.
    """
    if len(coord_min) == 2:  # 2D case
        x_min, y_min = coord_min
        x_max, y_max = coord_max
        # Draw filled rectangle first (optional)
        if fill:
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color=color, alpha=alpha)
            ax.add_patch(rect)
        # Overlay the border with proper width
        border_rect = plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor=color, linewidth=border_width
        )
        ax.add_patch(border_rect)
        return border_rect, f"Octant ID: {morton_code}\nMin: {coord_min}\nMax: {coord_max}"
    
    elif len(coord_min) == 3:  # 3D case
        x_min, y_min, z_min = coord_min
        x_max, y_max, z_max = coord_max
        # Define vertices of the bounding box
        vertices = [
            [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min)],  # Bottom face
            [(x_min, y_min, z_max), (x_max, y_min, z_max), (x_max, y_max, z_max), (x_min, y_max, z_max)],  # Top face
            [(x_min, y_min, z_min), (x_min, y_max, z_min), (x_min, y_max, z_max), (x_min, y_min, z_max)],  # Left face
            [(x_max, y_min, z_min), (x_max, y_max, z_min), (x_max, y_max, z_max), (x_max, y_min, z_max)],  # Right face
            [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_min, z_max), (x_min, y_min, z_max)],  # Front face
            [(x_min, y_max, z_min), (x_max, y_max, z_min), (x_max, y_max, z_max), (x_min, y_max, z_max)],  # Back face
        ]
        if fill:
            # Use Poly3DCollection to fill the faces of the 3D box
            poly3d_collection = Poly3DCollection(vertices, color=color, alpha=alpha, linewidths=border_width, edgecolors=color)
            ax.add_collection3d(poly3d_collection)
        # Draw edges with proper width if not filled
        for face in vertices:
            ax.plot(*zip(*face), color=color, linewidth=border_width)
        return None, f"Octant ID: {morton_code}\nMin: {coord_min}\nMax: {coord_max}"


def plot_octants(ax, octants, color, fill=True, alpha=0.1, border_color="black", border_width=1.0):
    """
    Plot octants for a single processor, with distinct borders and optional fill.
    """
    for octant in octants:
        # Draw filled regions first (optional)
        if fill:
            draw_bounds(ax, octant["coord_min"], octant["coord_max"], morton_code=octant["morton_code"], 
                        color=color, alpha=alpha, fill=True, border_width=border_width)
        
        # Overlay the borders to ensure they are visible
        draw_bounds(ax, octant["coord_min"], octant["coord_max"], morton_code=octant["morton_code"], 
                    color=border_color, alpha=1.0, fill=False, border_width=border_width)


def plot_particles(ax, particles, dim, color, particle_size=0.05):
    """
    Plot particles for a single processor.
    """
    if dim == 2:
        x_vals = [particle["coord"][0] for particle in particles]
        y_vals = [particle["coord"][1] for particle in particles]
        ax.scatter(x_vals, y_vals, color=color, s=particle_size)
    elif dim == 3:
        x_vals = [particle["coord"][0] for particle in particles]
        y_vals = [particle["coord"][1] for particle in particles]
        z_vals = [particle["coord"][2] for particle in particles]
        ax.scatter(x_vals, y_vals, z_vals, color=color, s=particle_size)

def plot_all_processors_in_subplots(data_folder, should_plot_particles=True, show_stats=True, title=None, max_depth=None, max_particles_per_octant=None):
    """
    Generate subplots for each processor, including optional statistics.
    Display overarching statistics at the top of the plot.
    """
    processor_data = read_data_for_all_processors(data_folder)
    num_processors = len(processor_data)
    cols = 2  # Define columns for the grid layout
    rows = (num_processors + 1) // cols  # Calculate required rows

    fig = plt.figure(figsize=(10, 5 * rows))

    # Stats variables
    total_particles = sum(len(particles) for _, particles, _ in processor_data)
    total_octants = sum(len(octants) for octants, _, _ in processor_data)

    for i, (octants, particles, dim) in enumerate(processor_data):
        print(f"Processor {i}: {len(octants)} octants, {len(particles)} particles, Dimension: {dim}D")  # Debug output

        ax = fig.add_subplot(rows, cols, i + 1, projection='3d' if dim == 3 else None)
        ax.set_title(f"Processor {i}")

        # Plot octants and particles
        plot_octants(ax, octants, color="blue", fill=False, alpha=0.1, border_color="black", border_width=0.5)
        if should_plot_particles:
            plot_particles(ax, particles, dim, color="red", particle_size=1)

        # Add stats per processor if enabled
        if show_stats:
            stats_text = f"Octants: {len(octants)}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

        # Configure plot axes
        setup_plot(ax, dim)

    # Add overarching stats at the top of the figure
    if show_stats:
        overall_stats_text = f"Total Particles: {total_particles}\nTotal Octants: {total_octants}"
        if max_depth is not None:
            overall_stats_text += f"\nMax Depth: {max_depth}"
        if max_particles_per_octant is not None:
            overall_stats_text += f"\nMax Particles/Octant: {max_particles_per_octant}"

        # Add overarching stats as a super title
        if title:
            overall_title = f"{title}\n{overall_stats_text}"
        else:
            overall_title = overall_stats_text

        fig.suptitle(overall_title, fontsize=14, fontweight='bold', y=0.98)

    # Adjust layout for the subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the overarching title
    plt.show()


def setup_plot(ax, dim):
    """
    Configure the plot's axes and labels.
    """
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if dim == 3:
        ax.set_zlabel("Z")

def plot_combined_processors(data_folder, should_plot_particles=False, show_stats=True, title=None, max_depth=None, max_particles_per_octant=None):
    """
    Combine data from all processors and plot them in a single plot with optional statistics.
    """
    processor_data = read_data_for_all_processors(data_folder)
    dim = 2  # Default to 2D if no particles are found

    # Define a color cycle for different processors
    colors = itertools.cycle(['blue', 'green', 'red', 'purple', 'orange', 
                               'brown', 'pink', 'gray', 'olive', 'cyan'])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d' if dim == 3 else None)
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Combined Data from All Processors")

    # Stats variables
    total_particles = 0
    total_octants = 0
    rank_stats = []

    # Keep track of colors and labels for the legend
    legend_handles = []

    for i, (octants, particles, processor_dim) in enumerate(processor_data):
        color = next(colors)  # Get the next color for this processor
        dim = max(dim, processor_dim)  # Update to 3D if any processor has 3D data

        # Plot octants and particles
        plot_octants(ax, octants, color=color, fill=True, alpha=0.25, border_color="black", border_width=0.5)
        if should_plot_particles:
            plot_particles(ax, particles, dim, color=color, particle_size=10)

        # Update stats
        num_octants = len(octants)
        num_particles = len(particles)
        total_octants += num_octants
        total_particles += num_particles
        rank_stats.append((i, num_octants, num_particles))

        # Add a legend handle for this processor
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                         markersize=10, label=f"Processor {i}"))

    setup_plot(ax, dim)

    # Add the legend
    ax.legend(handles=legend_handles, loc="upper right")

    # Display stats if enabled
    if show_stats:
        stats_text = f"Total Particles: {total_particles}\nTotal Octants: {total_octants}\n"
        if max_depth is not None:
            stats_text += f"Max Depth: {max_depth}\n"
        if max_particles_per_octant is not None:
            stats_text += f"Max Particles/Octant: {max_particles_per_octant}\n"
        stats_text += "\nPer Rank Stats:\n"
        for rank, num_octants, num_particles in rank_stats:
            stats_text += f"Rank {rank}: {num_particles} particles, {num_octants} octants\n"

        # Add stats as an annotation box
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

#########################################
#               OUTPUT                  #
#########################################
# (uncomment the one you dont need)     #
#########################################

data_folder = "output"

# those have to be set manualy by you, else we dont print them
show_stats=True

title=None
max_depth=None
max_particles_per_octant=None

"""
This will generate one plot, where each processor will be colored differently. (not that usefull)
"""
plot_combined_processors(data_folder=data_folder, show_stats=show_stats, title=title, max_depth=max_depth, max_particles_per_octant=max_particles_per_octant, should_plot_particles=False)

"""
This generates one plot per processor.
"""
#plot_all_processors_in_subplots(data_folder=data_folder, should_plot_particles=True, show_stats=show_stats, title=title, max_depth=max_depth, max_particles_per_octant=max_particles_per_octant)

