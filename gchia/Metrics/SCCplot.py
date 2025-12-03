import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.ticker import MaxNLocator
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from gchia.Config.PlotConfig import PlotConfig

def plot_scc(file_paths, resolution=10000, output_path=None,min_y=0):
    """
    Plot datasets from file paths with custom legends and beautiful colors.
    Parameters:
        file_paths: dict, mapping from file paths to legend names.
        resolution: int, genomic distance resolution in base pairs (default: 10,000).
        output_path: str, path to save the plot (optional).
    """
    

    fig, ax = PlotConfig.create_figure('square')
    

    
    
    # Generate colors based on the provided colormap
    
    color_index = 0  

    # Prepare data dictionary for later CSV file generation
    csv_data = {}
    max_length = 0
    max_distance = 0
    
    # Iterate through each file and its corresponding legend
    for (file_path, legend_name) in file_paths.items():
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping.")
            continue
        
        # Load data from the file
        data = np.load(file_path, allow_pickle=True)
        # data = np.nan_to_num(data, nan=0.0)
        
        # Compute genomic distances in kilobases (kb)
        distances = np.arange(len(data)) * resolution / 1000
        max_distance = max(max_distance, distances[-1] if len(distances) > 0 else 0)
        
        # Store data in dictionary for CSV generation
        csv_data[f'Distance'] = distances
        csv_data[legend_name] = data
        max_length = max(max_length, len(data))
        
        # Select color from COLOR_PALETTE, cycling back if necessary
        color = PlotConfig.get_color(color_index)
        ax.plot(distances, data, label=legend_name, 
               color=color)
        color_index += 1  # Increment color index
        
        
    # Configure plot labels, title, grid, and legend with custom font sizes
    ax.set_xlabel('Genomic Distance (kb)')
    ax.set_ylabel('Pearson Correlation Coefficient')
    ax.set_ylim(min_y, 1)
    # Set x-axis ticks, evenly divided into 4 parts (display 5 tick values)
    if max_distance > 0:
        xtick_positions = np.linspace(0, max_distance, 5)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([f'{int(x)}' if x == int(x) else f'{x:.1f}' 
                           for x in xtick_positions])
    
    
    PlotConfig.setup_legend(ax, loc='upper right')
    
    # Save the plot to a file or display it
    if output_path:
        PlotConfig.save_figure(fig, output_path, close=False)  # Increase DPI for higher quality output
        
        
        csv_output_path = os.path.splitext(output_path)[0] + '.csv'
        
        for key, value in csv_data.items():
            if len(value) < max_length:
                # Pad shorter arrays with NaN values
                padded_value = np.pad(value, (0, max_length - len(value)), 
                             mode='constant', constant_values=0)
                csv_data[key] = padded_value
        
            df = pd.DataFrame(csv_data)
            df = df.fillna(0)
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
    plot_scc(file_dict, 
                 fontsize=14, 
                 legend_fontsize=12, 
                 tick_fontsize=12, 
                 font_family='Arial',
                 font_weight='normal',
                 label_weight='bold',
                 legend_weight='normal')