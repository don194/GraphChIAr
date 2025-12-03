#!/usr/bin/env python3
"""
PlotConfig - Unified Plot Configuration System

A standardized plotting configuration system for generating consistent small plots
that can be perfectly assembled in Visio without any scaling adjustments.

Key Features:
- Standard figure sizes for perfect alignment
- Unified font and color schemes
- Precise margin control
- High-quality output (300 DPI + SVG)

Usage:
    from plot_config import PlotConfig
    
    fig, ax = PlotConfig.create_figure('standard')
    ax.plot(x, y, color=PlotConfig.get_color(0))
    PlotConfig.save_figure(fig, 'output.svg')
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

class PlotConfig:
    """Unified plotting configuration class"""
    
    # === Standard Figure Sizes (inches) ===
    SIZES = {
        # Basic sizes
        'standard': (3.0, 2.5),     # Standard small plot - for most charts
        'square': (2.5, 2.5),       # Square plot - for heatmaps, correlation matrices
        'wide': (4.0, 2.5),         # Wide plot - for time series, long labels
        'tall': (3.0, 3.5),         # Tall plot - for multi-series bar charts
        'mini': (2.5, 2.0),         # Mini plot - for simple displays
        'half': (8, 3.5),         # Half-width plot - for insets
        # Special sizes
        'hic_triangle': (3.5, 3.5), # Hi-C triangular heatmap
        'legend_wide': (4.5, 2.5),  # Wide plot with external legend space
    }
    
    # === Font Configuration ===
    FONT_FAMILY = 'Times New Roman'  # Default font family
    
    # Alternative fonts (fallback order)
    FONT_OPTIONS = [
        'Times New Roman',
        'serif',
        'DejaVu Serif',
        'Liberation Serif'
    ]
    
    FONTS = {
        'title': 18,        # Figure title
        'label': 14,        # Axis labels (xlabel, ylabel)
        'tick': 12,         # Tick labels
        'legend': 12,       # Legend text
        'annotation': 9,    # Value annotations, notes
        'colorbar': 12,     # Colorbar labels
    }
    
    # === Standard Color Palette ===
    COLORS = [
        # '#3B85BD',  # Light blue - primary series
        # '#9DB648',  # Teal - secondary series  
        '#1F77B4',  # Dark blue - tertiary series
        '#FF7F0E',  # Orange - quaternary series
        '#2CA02C',  # Green - fifth series
        '#D62728',  # Red - sixth series
        '#9467BD',  # Purple - seventh series
        '#8C564B',  # Brown - eighth series
    ]
    
    # Special purpose colors
    SPECIAL_COLORS = {
        'reference': '#20639B',     # Reference data - dark blue
        'prediction': '#FF7F0E',    # Prediction data - orange  
        'highlight': '#D62728',     # Highlight/emphasis - red
        'neutral': '#666666',       # Neutral color - gray
    }
    
    # === Margin Configuration ===
    MARGINS_INCHES = {
        'left': 0.8,       # Left margin - Y-axis label space (增加)
        'right': 0.1,      # Right margin
        'top': 0.3,        # Top margin - title space (增加，为数值标注留空间)  
        'bottom': 0.5,     # Bottom margin - X-axis label space
        
        # Special margin adjustments
        'rotated_labels': 0.8,     # Bottom margin with rotated X labels
        'external':3
    }
    # === Line and Marker Styles ===
    LINE_STYLES = {
        'width': 1.5,               # Standard line width
        'marker_size': 4,           # Standard marker size
        'alpha': 0.8,               # Standard transparency
    }
    
    MARKERS = ['o', 's', '^', 'D', 'x', '*', 'p', 'h']  # Standard marker styles
    
    # === Other Configuration ===
    DPI = 300                       # Standard resolution
    GRID_ALPHA = 0.3                # Grid transparency
    SAVE_FORMAT = 'svg'             # Default save format
    
    @classmethod
    def create_figure(cls, size_type: str = 'standard', 
                     has_colorbar: bool = False, 
                     has_external_legend: bool = False,
                     has_rotated_labels: bool = False) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a standardized figure with guaranteed plot area size
        
        Parameters:
        -----------
        size_type : str
            Size type from SIZES dictionary (defines plot area size)
        has_colorbar : bool
            Whether the plot has a colorbar (adjusts right margin)
        has_external_legend : bool  
            Whether the plot has an external legend (adjusts right margin)
        has_rotated_labels : bool
            Whether X-axis labels are rotated (adjusts bottom margin)
            
        Returns:
        --------
        fig, ax : matplotlib figure and axes objects
        """
        if size_type not in cls.SIZES:
            raise ValueError(f"Unsupported size type: {size_type}. Available: {list(cls.SIZES.keys())}")
        
        plot_width, plot_height = cls.SIZES[size_type]
        left_margin = cls.MARGINS_INCHES['left']
        right_margin = cls.MARGINS_INCHES['right']
        top_margin = cls.MARGINS_INCHES['top']
        bottom_margin = cls.MARGINS_INCHES['bottom']
        
        # Adjust width for special requirements
        if has_external_legend:
            right_margin += cls.MARGINS_INCHES['external'] # Add space for external legend
        elif has_colorbar:
            right_margin += cls.MARGINS_INCHES['external']  # Add space for colorbar
        if has_rotated_labels:
            bottom_margin = cls.MARGINS_INCHES['rotated_labels_bottom']
        total_width = plot_width + left_margin + right_margin
        total_height = plot_height + top_margin + bottom_margin
        # Create figure
        fig = plt.figure(figsize=(total_width, total_height), dpi=cls.DPI)
        
        # Setup global style
        cls._setup_style()
        left_ratio = left_margin / total_width
        bottom_ratio = bottom_margin / total_height
        width_ratio = plot_width / total_width
        height_ratio = plot_height / total_height
        
        # Create axes with precise positioning
        ax = fig.add_subplot(111)
        ax.set_position([left_ratio, bottom_ratio, width_ratio, height_ratio])
        
        return fig, ax
    
    @classmethod
    def _setup_style(cls):
        """Setup global matplotlib style"""
        # Set font family with fallback
        cls._set_font()
        
        plt.rcParams.update({
            # Font settings
            'font.family': cls.FONT_FAMILY,
            'font.size': cls.FONTS['tick'],
            'axes.labelsize': cls.FONTS['label'],
            'axes.titlesize': cls.FONTS['title'],
            'xtick.labelsize': cls.FONTS['tick'],
            'ytick.labelsize': cls.FONTS['tick'],
            'legend.fontsize': cls.FONTS['legend'],
            
            # Tick settings
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
            
            # Axes settings  
            'axes.linewidth': 0.8,
            'axes.edgecolor': '#000000',
            
            # Legend settings
            'legend.frameon': False,
            'legend.numpoints': 1,
        })

    @classmethod
    def _set_font(cls):
        """Set font family with fallback options"""
        import matplotlib.font_manager as fm
        
        # Get available fonts
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # Find first available font from options
        for font in cls.FONT_OPTIONS:
            if font in available_fonts or font in ['serif', 'sans-serif', 'monospace']:
                cls.FONT_FAMILY = font
                print(f"Using font: {font}")
                return
        
        # Fallback to default if none found
        cls.FONT_FAMILY = 'serif'
        print(f"Using fallback font: serif")

    @classmethod
    def set_font_family(cls, font_family: str):
        """
        Set custom font family
        
        Parameters:
        -----------
        font_family : str
            Font family name (e.g., 'Times New Roman', 'Arial', 'serif')
        """
        cls.FONT_FAMILY = font_family
        # Update matplotlib rcParams
        plt.rcParams['font.family'] = font_family
        print(f"Font family set to: {font_family}")

    @classmethod
    def list_available_fonts(cls):
        """List all available fonts on the system"""
        import matplotlib.font_manager as fm
        
        available_fonts = sorted(set([f.name for f in fm.fontManager.ttflist]))
        print("Available fonts:")
        for i, font in enumerate(available_fonts, 1):
            print(f"  {i:3d}. {font}")
        return available_fonts
    
    @classmethod
    def save_figure(cls, fig: plt.Figure, filename: str, 
                   format: str = None, close: bool = True) -> None:
        """
        Save standardized figure
        
        Parameters:
        -----------
        fig : plt.Figure
            Figure to save
        filename : str
            Filename (including path)
        format : str
            Save format, defaults to SAVE_FORMAT
        close : bool
            Whether to close figure after saving
        """
        # Create directory if needed
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format
        if format is None:
            format = cls.SAVE_FORMAT
        
        # Save figure with precise margins (no additional bbox adjustment)
        fig.savefig(filename, 
                   format=format, 
                   dpi=cls.DPI,
                   bbox_inches=None,  # Use preset margins, no extra adjustment
                   pad_inches=0)
        
        print(f"Figure saved: {filename}")
        
        if close:
            plt.close(fig)
    
    @classmethod
    def get_color(cls, index: int) -> str:
        """
        Get standard color by index
        
        Parameters:
        -----------
        index : int
            Color index
            
        Returns:
        --------
        str : Color code
        """
        return cls.COLORS[index % len(cls.COLORS)]
    
    @classmethod
    def get_marker(cls, index: int) -> str:
        """Get standard marker style by index"""
        return cls.MARKERS[index % len(cls.MARKERS)]
    
    @classmethod
    def setup_grid(cls, ax: plt.Axes, show: bool = True) -> None:
        """Setup standard grid"""
        if show:
            ax.grid(True, linestyle='--', alpha=cls.GRID_ALPHA, linewidth=0.5)
    
    @classmethod
    def setup_legend(cls, ax: plt.Axes, loc: str = 'best', 
                    outside: bool = False) -> None:
        """Setup standard legend"""
        if outside:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.legend(loc=loc)
    
    @classmethod
    def add_value_labels(cls, ax: plt.Axes, bars, values: List, 
                        format_str: str = '{:.0f}', offset_ratio: float = 0.02):
        """
        Add value labels to bar chart
        
        Parameters:
        -----------
        ax : plt.Axes
            Axes object
        bars : matplotlib bar container
            Bar chart object
        values : List
            List of values
        format_str : str
            Value formatting string
        offset_ratio : float
            Label offset ratio
        """
        max_height = max([bar.get_height() for bar in bars])
        
        # Adjust y-axis limit to accommodate labels
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], max_height * 1.15)  # Add 15% space for labels
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + max_height * offset_ratio,
                   format_str.format(val),
                   ha='center', va='bottom',
                   fontsize=cls.FONTS['annotation'])


# === Usage Examples ===

def example_line_plot():
    """Example: How to create a standardized line plot"""
    
    # Generate sample data
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create standardized figure
    fig, ax = PlotConfig.create_figure('square')
    
    # Plot data with standard colors and styles
    ax.plot(x, y1, label='Sin(x)', 
           color=PlotConfig.get_color(0), 
           linewidth=PlotConfig.LINE_STYLES['width'],
           marker=PlotConfig.get_marker(0),
           markersize=PlotConfig.LINE_STYLES['marker_size'])
    
    ax.plot(x, y2, label='Cos(x)',
           color=PlotConfig.get_color(1),
           linewidth=PlotConfig.LINE_STYLES['width'],
           marker=PlotConfig.get_marker(1), 
           markersize=PlotConfig.LINE_STYLES['marker_size'])
    
    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis') 
    ax.set_title('Example Line Plot')
    
    # Setup grid and legend
    PlotConfig.setup_grid(ax)
    PlotConfig.setup_legend(ax)
    
    # Save figure
    PlotConfig.save_figure(fig, 'example_line_plot.svg')

def example_bar_plot():
    """Example: How to create a standardized bar plot"""
    
    # Generate sample data
    categories = ['A', 'B', 'C', 'D']
    values = [10, 25, 15, 30]
    
    # Create figure
    fig, ax = PlotConfig.create_figure('square')
    
    # Plot bar chart with standard colors
    colors = [PlotConfig.get_color(i) for i in range(len(values))]
    bars = ax.bar(categories, values, color=colors)
    
    # Add value labels
    PlotConfig.add_value_labels(ax, bars, values)
    
    # Set labels
    ax.set_xlabel('Category')
    ax.set_ylabel('Value')
    ax.set_title('Example Bar Plot')
    
    # Save figure
    PlotConfig.save_figure(fig, 'example_bar_plot.svg')

if __name__ == "__main__":
    print("PlotConfig - Unified Plot Configuration System")
    print("=" * 50)
    print()
    print("Available figure sizes:")
    for size_type, dimensions in PlotConfig.SIZES.items():
        print(f"  {size_type}: {dimensions[0]}×{dimensions[1]} inches")
    print()
    print("Standard colors:")
    for i, color in enumerate(PlotConfig.COLORS):
        print(f"  Index {i}: {color}")
    print()
    
    # Generate examples
    print("Generating example plots...")
    example_line_plot()
    example_bar_plot()
    print("Example plots generated!")