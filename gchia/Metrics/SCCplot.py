import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.ticker import MaxNLocator
COLOR_PALETTE = ['#6EC5E9', '#3CAEA3', '#20639B','#FF7F0E' ]
def plot_datasets(file_paths, resolution=10000, output_path=None, fontsize=20, legend_fontsize=18, 
                  tick_fontsize=20, font_family=None, font_weight='normal', label_weight='normal',
                  legend_weight='normal'):
    """
    Plot datasets from file paths with custom legends and beautiful colors.
    Parameters:
        file_paths: dict, mapping from file paths to legend names.
        resolution: int, genomic distance resolution in base pairs (default: 10,000).
        output_path: str, path to save the plot (optional).
        fontsize: int, font size for axis labels (default: 16).
        legend_fontsize: int, font size for legend text (default: 14).
        tick_fontsize: int, font size for tick labels (default: 16).
        font_family: str, font family for all text elements (default: 'Arial')
        font_weight: str, default font weight for all text (default: 'normal')
        label_weight: str, font weight for axis labels (default: 'bold')
        legend_weight: str, font weight for legend text (default: 'normal')
    """
    # Set global font
    if font_family is not None:
        plt.rcParams['font.family'] = font_family
    plt.rcParams['font.weight'] = font_weight
    
    # Set ticks to point inward
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    plt.figure(figsize=(5, 5))
    
    # Generate colors based on the provided colormap
    num_colors = len(file_paths)
    color_index = 2  # Start from the second color in COLOR_PALETTE

    # Prepare data dictionary for later CSV file generation
    csv_data = {}
    max_length = 0
    max_distance = 0
    
    # Iterate through each file and its corresponding legend
    for idx, (file_path, legend_name) in enumerate(file_paths.items()):
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping.")
            continue
        
        # Load data from the file
        data = np.load(file_path, allow_pickle=True)
        
        # Compute genomic distances in kilobases (kb)
        distances = np.arange(len(data)) * resolution / 1000
        max_distance = max(max_distance, distances[-1] if len(distances) > 0 else 0)
        
        # Store data in dictionary for CSV generation
        csv_data[f'Distance'] = distances
        csv_data[legend_name] = data
        max_length = max(max_length, len(data))
        
        # Select color from COLOR_PALETTE, cycling back if necessary
        color = COLOR_PALETTE[color_index % len(COLOR_PALETTE)]
        color_index += 1  # Increment color index
        
        # Plot the data with a unique color and corresponding legend
        plt.plot(distances, data, label=legend_name, linewidth=2, color=color)
    
    # Configure plot labels, title, grid, and legend with custom font sizes
    plt.xlabel('Genomic Distance (kb)', fontsize=fontsize, fontfamily=font_family, fontweight=label_weight)
    plt.ylabel('Pearson Correlation Coefficient', fontsize=fontsize, fontfamily=font_family, fontweight=label_weight)
    # plt.title('Correlation vs Genomic Distance', fontsize=fontsize+4, fontfamily=font_family, fontweight='bold')
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)  # Limit y-axis range to [0, 1]
    
    # Get current axis object and set ticks to point inward
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in')
    
    # Set x-axis ticks, evenly divided into 4 parts (display 5 tick values)
    if max_distance > 0:
        # Calculate evenly spaced tick positions
        xtick_positions = np.linspace(0, max_distance, 5)
        # Set tick labels and font size
        plt.xticks(xtick_positions, [f'{int(x)}' if x == int(x) else f'{x:.1f}' for x in xtick_positions], 
                  fontsize=tick_fontsize, fontweight=font_weight)
    else:
        plt.xticks(fontsize=tick_fontsize, fontweight=font_weight)
    
    plt.yticks(fontsize=tick_fontsize, fontweight=font_weight)
    
    # Add legend with custom font size and weight
    plt.legend(loc='upper right', fontsize=legend_fontsize, frameon=False)
    plt.tight_layout()
    
    # Save the plot to a file or display it
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)  # Increase DPI for higher quality output
        print(f"Plot saved to {output_path}")
        
        csv_output_path = os.path.splitext(output_path)[0] + '.csv'
        
        for key, value in csv_data.items():
            if len(value) < max_length:
                # Pad shorter arrays with NaN values
                padded_value = np.pad(value, (0, max_length - len(value)), 
                                     mode='constant', constant_values=np.nan)
                csv_data[key] = padded_value
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_output_path, index=False)
        print(f"CSV data saved to {csv_output_path}")
    
    plt.show()
    
    plt.close()

if __name__ == '__main__':
    # Example usage
    file_dict = {
        'file1.npy': 'Dataset 1',
        'file2.npy': 'Dataset 2'
    }
    # Use custom font sizes, font family and font weights
    plot_datasets(file_dict, 
                 fontsize=14, 
                 legend_fontsize=12, 
                 tick_fontsize=12, 
                 font_family='Arial',
                 font_weight='normal',
                 label_weight='bold',
                 legend_weight='normal')