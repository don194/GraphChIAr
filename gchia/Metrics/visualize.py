import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import hicstraw

def visualize_multiple_HiC_with_epigenetics(HiCs, epis=None, fig_width=6,
                                          vmin=0, vmax=None, cmap='Reds', colorbar=True,
                                          colorbar_orientation='vertical',
                                          hic_labels=None, epi_labels=None, x_ticks=None, fontsize=36,
                                          label_fontsize=20, tick_fontsize=14, colorbar_fontsize=16,
                                          epi_colors=None, epi_yaxis=True,
                                          heatmap_ratio=0.6, epi_ratio=0.1,
                                          interval_between_hic=0.1,
                                          interval_after_hic_block=0.1,
                                          interval_between_epi=0.01,
                                          maxperc=99.5,
                                          output=None):   
    """
    Visualize multiple Hi-C contact maps in one figure with epigenetic signals at the bottom
    
    Args:
        HiCs (list): List of Hi-C contact maps, each is a numpy.array. Only upper triangles are used.
        epis (list): Epigenetic signals to be shown at the bottom. Default: None
        fig_width (float): the width of the figure. Then the height will be automatically calculated. Default: 18.0
        vmin (float): min value of the colormap. Default: 0
        vmax (float or list): max value of the colormap. Will use the max value in Hi-C data if not specified.
                             Can be a list for per-heatmap vmax settings.
        cmap (str or plt.cm or list): which colormap to use. Can be a list for different Hi-C maps.
        colorbar (bool): whether to add colorbar for the heatmap. Default: True
        colorbar_orientation (str): "horizontal" or "vertical". Default: "vertical"
        hic_labels (list): labels for each Hi-C map. Default: None
        epi_labels (list): the names of epigenetic marks. If None, there will be no labels at y axis.
        x_ticks (list or list of lists): a list of strings for x-axis ticks.
                                       Can be a list of lists for different Hi-C blocks.
        fontsize (int): general font size. Default: 36
        label_fontsize (int): font size for labels. Default: 36
        tick_fontsize (int): font size for tick labels. Default: 32
        colorbar_fontsize (int): font size for colorbar. Default: 28
        epi_colors (list): colors of epigenetic signals
        epi_yaxis (bool): whether add y-axis to epigenetic signals. Default: True
        heatmap_ratio (float): the ratio of (heatmap height) and (figure width). Default: 0.6
        epi_ratio (float): the ratio of (1D epi signal height) and (figure width). Default: 0.1
        interval_between_hic (float): the ratio of (interval between Hi-C blocks) and (figure width). Default: 0.1
        interval_after_hic_block (float): the ratio of (interval between Hi-C block and epigenetic signals) and (figure width). Default: 0.1
        interval_between_epi (float): the ratio of (interval between 1D signals) and (figure width). Default: 0.01

    No return. Display a figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    # Check inputs
    n_hic = len(HiCs)
    
    # If no epigenetic signals provided, create empty list
    if epis is None:
        epis = []
    n_epi = len(epis)
    
    # Handle vmax as a list or single value
    if vmax is None:
        vmax = [np.percentile(hic, maxperc) for hic in HiCs]
    elif not isinstance(vmax, list):
        vmax = [vmax] * n_hic
        
    # Handle cmap as a list or single value
    if not isinstance(cmap, list):
        cmap = [cmap] * n_hic
        
    # Get the size of each Hi-C matrix
    sizes = [len(hic) for hic in HiCs]
    
    # Check if all Hi-C matrices and epigenetic signals have the same size
    if n_epi > 0:
        for epi in epis:
            if len(epi) != sizes[0]:  # Compare against first Hi-C size
                raise ValueError('Epigenetic signal length must match Hi-C matrix size!')
    
    # Define the space for each row
    # First, for Hi-C heatmaps with intervals
    hic_ratios = []
    for i in range(n_hic):
        hic_ratios.append(heatmap_ratio)
        if i < n_hic - 1:  # Add interval between Hi-C blocks
            hic_ratios.append(interval_between_hic)
    
    # Add interval after Hi-C blocks
    if n_epi > 0:
        hic_ratios.append(interval_after_hic_block)
    
    # Then, for epigenetic signals with intervals
    epi_ratios = []
    for i in range(n_epi):
        epi_ratios.append(epi_ratio)
        if i < n_epi - 1:  # Add interval between epigenetic signals
            # 修复: 这里如果interval_between_epi为0或非常小，我们应该避免添加额外空间
            # 只有当间隔值大于某个最小阈值时才添加
            if interval_between_epi > 0.001:
                epi_ratios.append(interval_between_epi)
    
    # Combine all ratios
    full_ratios = np.array(hic_ratios + epi_ratios)
    
    # Calculate figure height
    fig_height = fig_width * np.sum(full_ratios)
    full_ratios = full_ratios / np.sum(full_ratios)  # normalize to 1 (ratios)
    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Split the figure into rows with different heights
    gs = GridSpec(len(full_ratios), 1, height_ratios=full_ratios)
    
    # Track current row in the grid
    current_row = 0
    
    # Plot each Hi-C block
    for i, hic in enumerate(HiCs):
        N = sizes[i]
        
        # Plot Hi-C heatmap
        ax_hic = plt.subplot(gs[current_row, :])
        
        # Define the rotated axes and coordinates for Hi-C
        coordinate = np.array([[[(x + y) / 2, y - x] for y in range(N + 1)] for x in range(N + 1)])
        X, Y = coordinate[:, :, 0], coordinate[:, :, 1]
        
        # Plot the heatmap
        im = ax_hic.pcolormesh(X, Y, hic, vmin=vmin, vmax=vmax[i], cmap=cmap[i])
        ax_hic.axis('off')
        ax_hic.set_ylim([0, N])
        ax_hic.set_xlim([0, N])
        
        # Add Hi-C label at the bottom of the heatmap instead of the title
        if hic_labels and i < len(hic_labels):
            # Position the text at the bottom of the heatmap
            ax_hic.text(N/2, -N*0.05, hic_labels[i], 
                       horizontalalignment='center', 
                       verticalalignment='top', 
                       fontsize=label_fontsize, 
                       fontweight='normal'
                       )
        
        # Add colorbar if requested for the first heatmap
        if colorbar and i == 0:
            if colorbar_orientation == 'horizontal':
                _left, _width, _bottom, _height = 0.12, 0.25, 1 - full_ratios[0] * 0.25, full_ratios[0] * 0.03
            elif colorbar_orientation == 'vertical':
                _left, _width, _bottom, _height = 0.9, 0.02, 1 - full_ratios[0] * 0.7, full_ratios[0] * 0.5
            else:
                raise ValueError('Wrong orientation!')
            
            # Calculate position in normalized figure coordinates
            cbar = plt.colorbar(im, cax=fig.add_axes([_left, _bottom, _width, _height]),
                                orientation=colorbar_orientation)
            cbar.ax.tick_params(labelsize=colorbar_fontsize)
            cbar.outline.set_visible(False)
        
        # Move to next row, including interval between Hi-C maps
        current_row += 1
        if i < n_hic - 1:
            current_row += 1
    
    # Skip interval after Hi-C blocks if there are epigenetic signals
    if n_epi > 0:
        current_row += 1
    
    # Plot epigenetic signals
    epi_row_indices = []  # 记录每个epi信号对应的行索引
    
    for i, epi in enumerate(epis):
        epi_row_indices.append(current_row)
        ax_epi = plt.subplot(gs[current_row, :])
        
        if epi_colors and i < len(epi_colors):
            ax_epi.fill_between(np.arange(N), 0, epi, color=epi_colors[i])
        else:
            ax_epi.fill_between(np.arange(N), 0, epi)
        
        ax_epi.spines['left'].set_visible(False)
        ax_epi.spines['right'].set_visible(False)
        ax_epi.spines['top'].set_visible(False)
        ax_epi.spines['bottom'].set_visible(False)
        
        if not epi_yaxis:
            ax_epi.set_yticks([])
            ax_epi.set_yticklabels([])
        else:
            ax_epi.spines['right'].set_visible(True)
            ax_epi.tick_params(labelsize=tick_fontsize)
            ax_epi.yaxis.tick_right()
        
        # Only show x-ticks on the last signal
        is_last_epi = (i == n_epi - 1)
        
        if not is_last_epi:
            ax_epi.set_xticks([])
            ax_epi.set_xticklabels([])
        
        ax_epi.set_xlim([-0.5, N - 0.5])
        
        # 修改: 将标签放在左侧
        if epi_labels and i < len(epi_labels):
            ax_epi.set_ylabel(epi_labels[i], fontsize=label_fontsize, rotation=0, fontweight='normal')
            # 将标签放在Y轴左侧，并调整位置
            ax_epi.yaxis.set_label_coords(-0.15, 0.5)
        
        # Add x-ticks if this is the last epigenetic track
        if is_last_epi:
            ax_epi.spines['bottom'].set_visible(True)
            
            if x_ticks:
                if isinstance(x_ticks[0], list) if x_ticks and len(x_ticks) > 0 else False:  # x_ticks is a list of lists
                    # Use first set of ticks for epigenetic signals
                    block_x_ticks = x_ticks[0] if x_ticks else None
                else:  # x_ticks is a single list
                    block_x_ticks = x_ticks
                
                if block_x_ticks:
                    tick_pos = np.linspace(0, N - 1, len(block_x_ticks))
                    ax_epi.set_xticks(tick_pos)
                    ax_epi.set_xticklabels(block_x_ticks, fontsize=tick_fontsize)
                    # 增大x轴刻度标签的大小
                    ax_epi.tick_params(axis='x', labelsize=tick_fontsize)
                    ax_epi.tick_params(axis='x', which='both', length=0)
            else:
                ax_epi.set_xticks([])
                ax_epi.set_xticklabels([])
        
        # Move to next row, including interval between epigenetic signals
        current_row += 1
        if i < n_epi - 1 and interval_between_epi > 0.001:  # 只有当间隔较大时才增加行
            current_row += 1
    
    # 调整所有epi信号的垂直比例，使其更靠近
    if n_epi > 1:
        # 获取所有epi信号的轴对象
        epi_axes = [plt.subplot(gs[idx]) for idx in epi_row_indices]
        
        # 调整所有epi信号的Y轴限制，使其更加一致
        for ax in epi_axes:
            # 获取当前Y轴限制
            y_min, y_max = ax.get_ylim()
            # 设置一个一致的Y轴高度比例以使信号显示相似
            ax.set_ylim(0, y_max * 1.05)  # 为信号顶部留出5%的空间
    
    plt.tight_layout()
    if output != None:
        plt.savefig(output, bbox_inches='tight',dpi=300)  # 增加DPI以获得更高质量的输出
        print(f"Plot saved to {output}")
    plt.show()
    plt.close()


