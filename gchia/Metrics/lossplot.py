import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import numpy as np
import matplotlib as mpl
COLOR_PALETTE = ['#6EC5E9', '#3CAEA3', '#20639B','#FF7F0E' ]

def plot_val_loss(file_dict, y_limit=None, smoothing_window=None, smoothing_polyorder=2, max_epoch=80, output_path=None, 
                 fontsize=20, title_fontsize=14, legend_fontsize=18, tick_fontsize=20, font_family=None):
    """
    Plots val_loss vs. epoch for multiple files with custom legend labels and optional smoothing.

    Parameters:
    - file_dict: A dictionary where the keys are file paths and the values are custom legend labels.
    - y_limit: The upper limit for the y-axis. If None, the default behavior is used.
    - smoothing_window: The window size for Savitzky-Golay filter to smooth the curve. If None, no smoothing is applied.
    - smoothing_polyorder: The polynomial order for the Savitzky-Golay filter.
    - max_epoch: Maximum epoch to plot (default: 80)
    - output_path: Path to save the output plot and CSV data (default: None)
    - fontsize: Font size for axis labels (default: 16)
    - title_fontsize: Font size for plot title (default: 14)
    - legend_fontsize: Font size for legend text (default: 14)
    - tick_fontsize: Font size for tick labels (default: 16)
    - font_family: Font family for all text elements (default: 'Arial')
    """
    # 设置全局字体
    if font_family is not None:
        plt.rcParams['font.family'] = font_family
    
    plt.figure(figsize=(5, 5))  # Set figure size
    color_cycle = COLOR_PALETTE # Automatically use different colors for each plot
    color_idx = 2  # Index to pick different colors from the color cycle
    
    # 设置刻度朝内
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    # Prepare data dictionary for generating CSV file later
    csv_data = {'Epoch': []}
    max_epochs = 0
    
    for file, legend_name in file_dict.items():
        try:
            # Read the data from the CSV file
            data = pd.read_csv(file)
            
            # Select relevant columns and drop rows with missing values
            data = data[['epoch', 'val_loss']].dropna()
            data = data[data['epoch'] < max_epoch]
            # epoch + 1 because the first epoch is 0
            data['epoch'] = data['epoch'] + 1
            
            # Update maximum epoch value for CSV data alignment
            max_epochs = max(max_epochs, len(data))
            
            # If smoothing_window is provided, apply Savitzky-Golay filter
            if smoothing_window is not None:
                smoothed_val_loss = savgol_filter(data['val_loss'][2:], window_length=smoothing_window, polyorder=smoothing_polyorder)
                smoothed_val_loss = pd.Series(smoothed_val_loss, index=data['epoch'][2:])  # Ensure correct index alignment
                
                # Concatenate the first two values with the smoothed values, reset index properly
                smoothed_val_loss_full = pd.concat([data[['epoch', 'val_loss']].iloc[:2].set_index('epoch')['val_loss'], smoothed_val_loss])
                
                # Plot the original data with transparency and same label and color
                plt.plot(data['epoch'], data['val_loss'], alpha=0.2, color=color_cycle[color_idx])
                
                # Plot the smoothed data with the same label and color
                plt.plot(smoothed_val_loss_full.index, smoothed_val_loss_full, label=legend_name, color=color_cycle[color_idx])
                
                # Save smoothed data for CSV
                csv_data[f"{legend_name}_raw"] = data['val_loss'].values
                csv_data[f"{legend_name}_smoothed"] = smoothed_val_loss_full.values
                if 'Epoch' not in csv_data or len(csv_data['Epoch']) < len(data['epoch']):
                    csv_data['Epoch'] = data['epoch'].values
            else:
                # Plot original data if no smoothing is applied
                plt.plot(data['epoch'], data['val_loss'], label=legend_name, color=color_cycle[color_idx])
                
                # Save original data for CSV
                csv_data[f"{legend_name}"] = data['val_loss'].values
                if 'Epoch' not in csv_data or len(csv_data['Epoch']) < len(data['epoch']):
                    csv_data['Epoch'] = data['epoch'].values
            
            # Move to the next color in the color cycle
            color_idx = (color_idx + 1) % len(color_cycle)
        except Exception as e:
            # Handle errors during file reading or processing
            print(f"Error processing file {file}: {e}")
    
    # Set labels and title with custom font sizes
    plt.xlabel('Epoch', fontsize=fontsize, fontfamily=font_family)
    plt.ylabel('Validation Loss', fontsize=fontsize, fontfamily=font_family)
    # plt.title('Validation Loss vs. Epoch', fontsize=title_fontsize, fontfamily=font_family)
    
    # 设置刻度值朝内
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in')
    
    # Set tick font sizes
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    
    # Set y-axis limit if provided
    if y_limit is not None:
        plt.ylim(top=y_limit)
    
    # Add legend with a title and set its position with custom font size
    plt.legend(loc='upper right', fontsize=legend_fontsize, frameon=False)
    # plt.grid(True)  # Enable grid
    
    # Save plot and CSV data
    if output_path is not None:
        # Save plot
        plt.savefig(output_path, bbox_inches='tight', dpi=300)  # Increased DPI for better resolution
        print(f"Plot saved to {output_path}")
        
        # Save CSV file
        csv_output_path = os.path.splitext(output_path)[0] + '.csv'
        
        # Ensure all data columns have consistent lengths
        df_data = {}
        for key, values in csv_data.items():
            if len(values) < max_epochs:
                # Pad shorter arrays with NaN
                padded_values = np.pad(values, (0, max_epochs - len(values)), 
                                      mode='constant', constant_values=np.nan)
                df_data[key] = padded_values
            else:
                df_data[key] = values
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(df_data)
        df.to_csv(csv_output_path, index=False)
        print(f"CSV data saved to {csv_output_path}")
    
    plt.show()  # Display the plot
    

if __name__ == '__main__':
    # Example usage of the function
    file_dict = {
        'file1.csv': 'Model 1',
        'file2.csv': 'Model 2',
        'file3.csv': 'Model 3'
    }
    # Example with custom font sizes and font family
    plot_val_loss(file_dict, y_limit=0.5, fontsize=14, legend_fontsize=12, tick_fontsize=12, font_family='Arial')
